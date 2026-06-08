"""
Incremental / extensible recursive sketches (reverse sweep).

A `SketchGroup` is one block-rank group of a TTStack sketch, stored as **raw (un-normalized)**
per-bond partial contractions in **global-sample order**: column-block `s` at bond `l` is sample
`s`, the chain `block(seed,1,s) ⊗ … ⊗ block(seed,N,s)` contracted with the vector's cores.
`extend_recursive_sketch!` grows a group to higher per-bond sample counts, computing only the
**deficient bonds, right-to-left, reusing the right neighbour's existing columns at the same global
positions** — so a cached sketch extends reproducibly across calls (history-independent).

Blocks are addressed by `(seed, bond, sample-index, reverse, group)`; the `group` namespace lets
several groups (e.g. a `block_rks` initial group and a `block_rks_inc` extension group) coexist with
independent draws. `finalize_cols` combines groups into the normalized sketch columns; the
`1/√count` normalization is applied only there, so the stored raw partials need no rescaling on reuse.

Invariant: per-bond counts are non-decreasing (`counts[k] ≤ counts[k+1]` for `k<N`; bond `N` recurses
against the tiled `ones` boundary) — required by the positional recursion.
"""
mutable struct SketchGroup{T}
  W::Vector{Array{T,3}}     # raw partials; W[l] :: (A.ttv_rks[l], brv[l], counts[l]); W[N+1]=ones(1,1,1)
  counts::Vector{Int}       # per-bond sample count; counts[N+1] = 1 (tiled boundary)
  brv::Vector{Int}          # block_rks_vec for this group's block_rks
  block_rks::Int
  group::Int                # seed namespace (disjoint draws across groups)
end

"""
    SketchGroup(T, A, block_rks, group) -> empty SketchGroup

Build an empty group (no samples yet) for vector `A`, with sketch element type derived from `T` and
`eltype(A)`. `block_rks` sets the per-bond block ranks (`brv = block_rks_vec`); `group` is the seed
namespace.
"""
function SketchGroup(::Type{T}, A::TTvector{TA,N}, block_rks::Int, group::Int) where {T<:Number,TA<:Number,N}
  TW = typeof(one(T)*one(TA))
  dims = A.ttv_dims
  brv = ones(Int, N+1); brv[1:N] .= block_rks
  for k = N:-1:1
    brv[k] = min(brv[k], dims[k]*brv[k+1])
  end
  W = Vector{Array{TW,3}}(undef, N+1)
  W[N+1] = ones(TW, 1, 1, 1)
  counts = zeros(Int, N+1); counts[N+1] = 1
  return SketchGroup{TW}(W, counts, brv, block_rks, group)
end

"""
    extend_recursive_sketch!(g, A, target; seed, orthogonal, timer) -> g

Grow group `g` so each bond `l` has at least `target[l]` samples, computing only the missing samples.
`target` must be non-decreasing (`target[k] ≤ target[k+1]`); samples are appended in global order.
"""
function extend_recursive_sketch!(g::SketchGroup{TW}, A::TTvector{TA,N}, target::AbstractVector{Int};
                                   seed::Int=1234, orthogonal::Bool=true,
                                   timer::TimerOutput=TimerOutput()) where {TW,TA,N}
  dims = A.ttv_dims
  @timeit timer "extend_recursive_sketch" begin
    # Size the per-call block / contraction buffers for the largest deficient bond, then reuse them
    # across bonds under GC.@preserve (as tt_recursive_sketch does) — avoids per-bond reallocation of
    # the block tensor and the contraction's gemm scratch.
    max_sketch = 0; max_contract = 0
    @inbounds for k = N:-1:1
      g.counts[k] >= target[k] && continue
      add = target[k] - g.counts[k]
      max_sketch = max(max_sketch, g.brv[k+1]*dims[k]*g.brv[k]*add)
      cb, = contract_sketch_core_backwards_batched_buffers_size(dims[k], A.ttv_rks[k], A.ttv_rks[k+1], g.brv[k+1], g.brv[k], add)
      max_contract = max(max_contract, cb)
    end
    max_sketch == 0 && return g                          # nothing deficient
    sketch_buffer   = Vector{TW}(undef, max_sketch)
    contract_buffer = (Vector{TW}(undef, max_contract),)
    GC.@preserve sketch_buffer contract_buffer begin
      @inbounds for k = N:-1:1
        g.counts[k] >= target[k] && continue
        # k<N recurses against the right neighbour's columns (needs them present); k=N recurses
        # against the tiled ones boundary, so target[N] is unconstrained.
        @assert k == N || target[k] <= g.counts[k+1] "non-decreasing counts required for the reverse recursion: bond $k target $(target[k]) exceeds right-neighbour count $(g.counts[k+1])"
        off = g.counts[k]; add = target[k] - off
        # Blocks at global sample indices off+1..off+add at bond k (group namespace g.group).
        Bβzbp = generate_sketch_blocks(seed, k, off, TW, g.brv[k+1], dims[k], g.brv[k], add, orthogonal;
                                       reverse=true, group=g.group, buffer=sketch_buffer, timer=timer)
        S = permutedims(Bβzbp, (2,1,3,4))           # (z, β, b, add) layout the kernel wants
        Wk_new = zeros(TW, A.ttv_rks[k], g.brv[k], add)
        # Recurse against the right neighbour's EXISTING columns at the same global positions (a
        # contiguous slab view — no copy; the tiled ones boundary for k = N). This is what makes a
        # sample globally addressable.
        V = k < N ? view(g.W[k+1], :, :, off+1:target[k]) : repeat(g.W[N+1], 1, 1, add)
        contract_sketch_core_backwards!(Wk_new, A.ttv_vec[k], S, V; buffer=contract_buffer)
        # Geometric-growth append into a capacity buffer (amortized O(final width), avoids the
        # O(width²) cat blow-up). finalize_cols / _slab_var index only the used 1:counts[k] slabs;
        # spare capacity is never read.
        if off == 0
          g.W[k] = Wk_new
        else
          Wk = g.W[k]
          if size(Wk, 3) < target[k]
            newcap = max(2*size(Wk, 3), target[k])
            buf = Array{TW,3}(undef, size(Wk, 1), size(Wk, 2), newcap)
            copyto!(view(buf, :, :, 1:off), view(Wk, :, :, 1:off))
            Wk = buf; g.W[k] = buf
          end
          copyto!(view(Wk, :, :, off+1:target[k]), Wk_new)
        end
        g.counts[k] = target[k]
      end
    end
  end
  return g
end

# Per-group scalar weights w_g (Σ w_g = 1) at bond l from the per-group prefix sample counts `cnt`.
# Combining unbiased per-block estimators of unequal variance ⇒ inverse-variance (precision)
# weighting is the BLUE; :equal / :column are oblivious closed-form approximations (PLAN.md/Stage 6).
function _group_weights(groups, l::Int, cnt::AbstractVector{Int}, weighting::Symbol)
  if weighting === :equal
    raw = Float64[cnt[gi] for gi in eachindex(groups)]                          # ∝ samples → 1/√Σcnt
  elseif weighting === :column
    raw = Float64[cnt[gi]*groups[gi].brv[l] for gi in eachindex(groups)]        # ∝ columns
  elseif weighting === :precision
    # Provisional empirical inverse-variance: per-group sample variance of the leading per-sample
    # slab norms. Exact per-bond-residual form is finalized in Stage 6.
    raw = Float64[cnt[gi] == 0 ? 0.0 : cnt[gi] / max(_slab_var(groups[gi], l, cnt[gi]), eps()) for gi in eachindex(groups)]
  else
    error("unknown block-averaging weighting :$weighting (use :equal, :column, or :precision)")
  end
  s = sum(raw)
  return raw ./ s
end

function _slab_var(g::SketchGroup, l::Int, p::Int)
  p <= 1 && return 1.0
  W = g.W[l]
  e = [sum(abs2, @view W[:, :, s]) for s in 1:p]
  m = sum(e)/p
  return sum(x->(x-m)^2, e)/(p-1)
end

"""
    finalize_cols(groups, l, rks_l; weighting=:equal, nsamp=nothing) -> Matrix

Combine the groups' raw partials at bond `l` into the normalized sketch columns
`[√w₁ · group₁ | √w₂ · group₂ | …] · (per-group 1/√count)`, an unbiased isometry for any
`Σ w_g = 1`. `rks_l = A.ttv_rks[l]`. `nsamp[gi]` optionally caps each group to its leading samples
(a prefix), so several caches grown to different sizes can be combined at a common size.
"""
function finalize_cols(groups::AbstractVector{<:SketchGroup{T}}, l::Int, rks_l::Int;
                       weighting::Symbol=:equal, nsamp=nothing, out=nothing) where {T}
  cnt = nsamp === nothing ? [g.counts[l] for g in groups] : nsamp
  @assert all(cnt[gi] <= groups[gi].counts[l] for gi in eachindex(groups)) "finalize_cols: nsamp exceeds available samples"
  ws = _group_weights(groups, l, cnt, weighting)
  total = sum(cnt[gi]*groups[gi].brv[l] for gi in eachindex(groups))
  # Write into the caller's buffer when given (no allocation), else a fresh matrix.
  dest = out === nothing ? Matrix{T}(undef, rks_l, total) : out
  coff = 0
  for (gi, g) in enumerate(groups)
    cnt[gi] == 0 && continue
    bc = g.brv[l]*cnt[gi]
    # View the leading cnt[gi] slabs (g.W[l] may carry more samples and/or spare capacity).
    M = reshape(view(g.W[l], :, :, 1:cnt[gi]), rks_l, bc)
    # Divide (not multiply-by-reciprocal) so the single-group :equal case is bit-identical to
    # tt_recursive_sketch's `W ./= sqrt(count)`; fused broadcast write, no temporary.
    @views dest[:, coff+1:coff+bc] .= M ./ sqrt(cnt[gi] / ws[gi])
    coff += bc
  end
  return dest
end

# ── Per-vector cache: one block-rks group (uniform) or two (mixed block_rks/block_rks_inc) ──────
# The last group is the *active* (growable) one; earlier groups are frozen (the initial sketch).
struct CachedSketch{T}
  groups::Vector{SketchGroup{T}}
end

# Per-bond block counts for a uniform target rank `R` (the initial-sketch heuristic).
function _heuristic_p(R::Int, brv::Vector{Int}, N::Int)
  rks = fill(R, N+1); rks[N+1] = 1
  return compute_sketch_blocks_heuristic(rks, brv, N; reverse=true)
end

"""
    cached_sketch(T, A, block_rks, block_rks_inc, init_rank; seed, orthogonal, timer) -> CachedSketch

Build a reusable sketch cache for `A`: an initial (frozen) group of block rank `block_rks` sized to
the `init_rank` heuristic, plus — when `block_rks_inc != block_rks` — an empty extension group of
block rank `block_rks_inc`. When `block_rks_inc == block_rks` the single group both seeds and extends.
Grow it with `ensure_columns!`; read normalized columns with `sketch_matrix`.
"""
function cached_sketch(::Type{T}, A::TTvector{TA,N}, block_rks::Int, block_rks_inc::Int, init_rank::Int;
                       seed::Int=1234, orthogonal::Bool=true, timer::TimerOutput=TimerOutput()) where {T<:Number,TA<:Number,N}
  init = SketchGroup(T, A, block_rks, 0)
  extend_recursive_sketch!(init, A, _heuristic_p(init_rank, init.brv, N); seed=seed, orthogonal=orthogonal, timer=timer)
  TW = eltype(init.W[1])
  groups = block_rks_inc == block_rks ? SketchGroup{TW}[init] :
                                        SketchGroup{TW}[init, SketchGroup(T, A, block_rks_inc, 1)]
  return CachedSketch{TW}(groups)
end

"""
    ensure_columns!(cache, A, want; seed, orthogonal, timer) -> cache

Grow the cache's active group so each bond `l` has at least `want[l]` *total* sketch columns
(across all groups). The frozen groups are left untouched; only the missing samples of the active
group are computed (reusing what's already cached).
"""
function ensure_columns!(cache::CachedSketch, A::TTvector{TA,N}, want::AbstractVector{Int};
                         seed::Int=1234, orthogonal::Bool=true, timer::TimerOutput=TimerOutput()) where {TA,N}
  active = cache.groups[end]
  # columns already supplied by the frozen groups
  frozen_cols = zeros(Int, N+1)
  @inbounds for gi in 1:length(cache.groups)-1, l in 1:N+1
    frozen_cols[l] += cache.groups[gi].counts[l] * cache.groups[gi].brv[l]
  end
  # active-group sample target covering the remaining columns, then the smallest non-decreasing
  # target ≥ that (running max from the left) so the reverse recursion is satisfiable.
  target = copy(active.counts)
  @inbounds for l in 1:N
    need = max(0, want[l] - frozen_cols[l])
    target[l] = max(active.counts[l], cld(need, active.brv[l]))
  end
  @inbounds for l in 2:N
    target[l] = max(target[l], target[l-1])
  end
  extend_recursive_sketch!(active, A, target; seed=seed, orthogonal=orthogonal, timer=timer)
  return cache
end

"""
    sketch_matrix(cache, l, rks_l; weighting=:equal, nsamp=nothing) -> Matrix

Normalized sketch columns at bond `l` (combines all groups via `finalize_cols`). `nsamp[gi]`
optionally caps each group to its leading samples so caches grown to different sizes combine at a
common size.
"""
sketch_matrix(cache::CachedSketch, l::Int, rks_l::Int; weighting::Symbol=:equal, nsamp=nothing, out=nothing) =
  finalize_cols(cache.groups, l, rks_l; weighting=weighting, nsamp=nsamp, out=out)

"""
    remat_into!(Wj, l, cache, rks_l, nsamp; weighting) -> Wj[l]

Materialize bond `l` of `cache` (prefix `nsamp`) into a **reused capacity buffer** `Wj[l]`,
geometric-growing it only when the needed width exceeds capacity — avoids reallocating the sketch
matrix on every extension. Consumers must index columns within the materialized width (the spare
capacity past it holds stale data).
"""
function remat_into!(Wj::Vector{Matrix{T}}, l::Int, cache::CachedSketch, rks_l::Int,
                     nsamp=nothing; weighting::Symbol=:equal) where {T}
  cnt = nsamp === nothing ? [g.counts[l] for g in cache.groups] : nsamp
  cols = sum(cnt[gi]*cache.groups[gi].brv[l] for gi in eachindex(cache.groups))
  if !isassigned(Wj, l) || size(Wj[l], 1) != rks_l || size(Wj[l], 2) < cols
    newcap = (isassigned(Wj, l) && size(Wj[l], 1) == rks_l) ? max(2*size(Wj[l], 2), cols) : cols
    Wj[l] = Matrix{T}(undef, rks_l, newcap)
  end
  finalize_cols(cache.groups, l, rks_l; weighting=weighting, nsamp=nsamp, out=view(Wj[l], :, 1:cols))
  return Wj[l]
end
