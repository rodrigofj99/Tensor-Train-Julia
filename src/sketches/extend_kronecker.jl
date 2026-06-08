"""
Incremental / extensible recursive **Kronecker / Hadamard** sketches (reverse sweep): a sketch of the
element-wise product `y[1] ⊙ … ⊙ y[M]` over `M` TTvector factors. The Kronecker analogue of
`SketchGroup` — same global-sample addressing, raw (un-normalized) storage, and
deficient-bonds-right-to-left extension — but each per-bond partial carries one rank dimension per
factor, so `W[l] :: (y[1].ttv_rks[l], …, y[M].ttv_rks[l], brv[l], counts[l])` is `(M+2)`-D and the
per-sample contraction is `contract_sketch_core_kronecker_backwards!` (no batched variant). Boundary
`W[N+1] = ones(1×(M+2))`.

`finalize_cols_kronecker` returns the **2-D** normalized sketch `(∏ᵢ y[i].ttv_rks[l], total_cols)` —
the flattened form the Hadamard rounding overload already consumes — so reuse is a pure reshape. The
random blocks are shared with the vector group at the same `(seed, bond, sample, reverse, group)`, and
`1/√count` normalization is applied only in `finalize_cols_kronecker`.
"""
mutable struct KroneckerSketchGroup{T,D}
  W::Vector{Array{T,D}}     # raw partials; W[l] :: (rks₁[l],…,rks_M[l], brv[l], counts[l]); D=M+2; W[N+1]=ones(1×D)
  counts::Vector{Int}       # per-bond sample count; counts[N+1] = 1 (tiled boundary)
  brv::Vector{Int}          # block_rks_vec for this group's block_rks
  block_rks::Int
  group::Int                # seed namespace (disjoint draws across groups)
end

"""
    KroneckerSketchGroup(T, A, block_rks, group) -> empty KroneckerSketchGroup

Build an empty Kronecker group (no samples yet) for the sketch of `A[1] ⊙ … ⊙ A[M]`. `brv` is computed
from `block_rks` and the (shared) factor dimensions, identical to the vector group's so the blocks line up.
"""
function KroneckerSketchGroup(::Type{T}, A::NTuple{M,TTvector{TA,N}}, block_rks::Int, group::Int) where {T<:Number,TA<:Number,N,M}
  TW = typeof(one(T)*one(TA))
  dims = A[1].ttv_dims
  brv = ones(Int, N+1); brv[1:N] .= block_rks
  for k = N:-1:1
    brv[k] = min(brv[k], dims[k]*brv[k+1])
  end
  D = M + 2
  W = Vector{Array{TW,D}}(undef, N+1)
  W[N+1] = ones(TW, ntuple(i->1, D)...)
  counts = zeros(Int, N+1); counts[N+1] = 1
  return KroneckerSketchGroup{TW,D}(W, counts, brv, block_rks, group)
end

"""
    extend_kronecker_sketch!(g, A, target; seed, orthogonal, timer) -> g

Grow Kronecker group `g` so each bond `l` has at least `target[l]` samples, computing only the missing
samples (deficient bonds, right-to-left, reusing the right neighbour's existing columns at the same
global positions). `target` must be non-decreasing over `1:N`.
"""
function extend_kronecker_sketch!(g::KroneckerSketchGroup{TW,D}, A::NTuple{M,TTvector{TA,N}},
                                   target::AbstractVector{Int}; seed::Int=1234, orthogonal::Bool=true,
                                   timer::TimerOutput=TimerOutput()) where {TW,TA,N,M,D}
  dims = A[1].ttv_dims
  @timeit timer "extend_kronecker_sketch" begin
    # Size the per-call block / contraction buffers for the largest deficient bond, then reuse them
    # across bonds under GC.@preserve (as tt_recursive_sketch does).
    max_sketch = 0; cb1 = 0; cb2 = 0; cb3 = 0
    @inbounds for k = N:-1:1
      g.counts[k] >= target[k] && continue
      add = target[k] - g.counts[k]
      max_sketch = max(max_sketch, g.brv[k+1]*dims[k]*g.brv[k]*add)
      v = ntuple(i->(i==M+1 ? g.brv[k+1] : A[i].ttv_rks[k+1]), M+1)
      w = ntuple(i->(i==M+1 ? g.brv[k]   : A[i].ttv_rks[k]),   M+1)
      b1, b2, b3 = contract_sketch_core_backwards_kronecker_buffers_size(dims[k], v, w)
      cb1 = max(cb1, b1); cb2 = max(cb2, b2); cb3 = max(cb3, b3)
    end
    max_sketch == 0 && return g                          # nothing deficient
    sketch_buffer   = Vector{TW}(undef, max_sketch)
    contract_buffer = (Vector{TW}(undef, cb1), Vector{TW}(undef, cb2), Vector{TW}(undef, cb3))
    GC.@preserve sketch_buffer contract_buffer begin
      @inbounds for k = N:-1:1
        g.counts[k] >= target[k] && continue
        @assert k == N || target[k] <= g.counts[k+1] "non-decreasing counts required for the reverse recursion: bond $k target $(target[k]) exceeds right-neighbour count $(g.counts[k+1])"
        off = g.counts[k]; add = target[k] - off
        # Blocks at global sample indices off+1..off+add at bond k (group namespace g.group). The
        # block shape (brv[k+1], dims[k], brv[k]) already matches the kernel's S — no permute.
        B = generate_sketch_blocks(seed, k, off, TW, g.brv[k+1], dims[k], g.brv[k], add, orthogonal;
                                   reverse=true, group=g.group, buffer=sketch_buffer, timer=timer)
        Wk_new = zeros(TW, ntuple(i->A[i].ttv_rks[k], M)..., g.brv[k], add)
        cores = ntuple(i -> A[i].ttv_vec[k], M)
        for s in 1:add
          Vs = k < N ? view(g.W[k+1], ntuple(i->Colon(), M+1)..., off+s) :
                       view(g.W[N+1], ntuple(i->Colon(), M+1)..., 1)
          Wks = view(Wk_new, ntuple(i->Colon(), M+1)..., s)
          contract_sketch_core_kronecker_backwards!(Wks, cores, view(B, :, :, :, s), Vs; buffer=contract_buffer)
        end
        # Geometric-growth append into a capacity buffer (dim D = samples).
        if off == 0
          g.W[k] = Wk_new
        else
          Wk = g.W[k]
          if size(Wk, D) < target[k]
            newcap = max(2*size(Wk, D), target[k])
            buf = Array{TW,D}(undef, ntuple(i->size(Wk, i), D-1)..., newcap)
            copyto!(view(buf, ntuple(i->Colon(), D-1)..., 1:off), view(Wk, ntuple(i->Colon(), D-1)..., 1:off))
            Wk = buf; g.W[k] = buf
          end
          copyto!(view(Wk, ntuple(i->Colon(), D-1)..., off+1:target[k]), Wk_new)
        end
        g.counts[k] = target[k]
      end
    end
  end
  return g
end

# Per-group sample variance of the leading per-sample slab norms (Kronecker slab is W[…, s]).
function _slab_var(g::KroneckerSketchGroup{T,D}, l::Int, p::Int) where {T,D}
  p <= 1 && return 1.0
  W = g.W[l]
  e = [sum(abs2, view(W, ntuple(i->Colon(), D-1)..., s)) for s in 1:p]
  m = sum(e)/p
  return sum(x->(x-m)^2, e)/(p-1)
end

"""
    finalize_cols_kronecker(groups, l, prod_rks; weighting=:equal, nsamp=nothing, out=nothing) -> Matrix

Combine the Kronecker groups' raw partials at bond `l` into the normalized **2-D** sketch
`(prod_rks, total_cols)`, an unbiased isometry. `prod_rks = ∏ᵢ y[i].ttv_rks[l]`. The single-group
`:equal` case is bit-identical to `tt_recursive_sketch(y::NTuple)`'s reshaped/flattened `W[l]`.
"""
function finalize_cols_kronecker(groups::AbstractVector{<:KroneckerSketchGroup{T,D}}, l::Int, prod_rks::Int;
                                 weighting::Symbol=:equal, nsamp=nothing, out=nothing) where {T,D}
  cnt = nsamp === nothing ? [g.counts[l] for g in groups] : nsamp
  @assert all(cnt[gi] <= groups[gi].counts[l] for gi in eachindex(groups)) "finalize_cols_kronecker: nsamp exceeds available samples"
  ws = _group_weights(groups, l, cnt, weighting)
  total = sum(cnt[gi]*groups[gi].brv[l] for gi in eachindex(groups))
  dest = out === nothing ? Matrix{T}(undef, prod_rks, total) : out
  coff = 0
  for (gi, g) in enumerate(groups)
    cnt[gi] == 0 && continue
    bc = g.brv[l]*cnt[gi]
    # View the leading cnt[gi] slabs (a contiguous prefix of the last dim); reshape
    # (rks₁,…,rks_M, brv, cnt) → (prod_rks, brv*cnt), column-major matching the reference.
    M = reshape(view(g.W[l], ntuple(i->Colon(), D-1)..., 1:cnt[gi]), prod_rks, bc)
    @views dest[:, coff+1:coff+bc] .= M ./ sqrt(cnt[gi] / ws[gi])
    coff += bc
  end
  return dest
end

# ── Per-tuple Kronecker cache: one block-rks group (uniform) or two (mixed block_rks/block_rks_inc) ──
struct KroneckerCachedSketch{T,D}
  groups::Vector{KroneckerSketchGroup{T,D}}
end

"""
    cached_kronecker_sketch(T, A, block_rks, block_rks_inc, init_rank; seed, orthogonal, timer) -> KroneckerCachedSketch

Build a reusable Kronecker sketch cache for `A[1] ⊙ … ⊙ A[M]`: an initial (frozen) group of block
rank `block_rks` sized to the `init_rank` heuristic, plus — when `block_rks_inc != block_rks` — an
empty extension group of block rank `block_rks_inc`. Grow with `ensure_columns!`; read with `sketch_matrix`.
"""
function cached_kronecker_sketch(::Type{T}, A::NTuple{M,TTvector{TA,N}},
                                 block_rks::Int, block_rks_inc::Int, init_rank::Int;
                                 seed::Int=1234, orthogonal::Bool=true, timer::TimerOutput=TimerOutput()) where {T<:Number,TA<:Number,N,M}
  init = KroneckerSketchGroup(T, A, block_rks, 0)
  extend_kronecker_sketch!(init, A, _heuristic_p(init_rank, init.brv, N); seed=seed, orthogonal=orthogonal, timer=timer)
  D = M + 2
  TW = eltype(init.W[N+1])
  groups = block_rks_inc == block_rks ? KroneckerSketchGroup{TW,D}[init] :
                                        KroneckerSketchGroup{TW,D}[init, KroneckerSketchGroup(T, A, block_rks_inc, 1)]
  return KroneckerCachedSketch{TW,D}(groups)
end

"""
    ensure_columns!(cache::KroneckerCachedSketch, A, want; seed, orthogonal, timer) -> cache

Grow the cache's active group so each bond `l` has at least `want[l]` *total* sketch columns (across
all groups). Frozen groups untouched; only the missing samples of the active group are computed.
"""
function ensure_columns!(cache::KroneckerCachedSketch, A::NTuple{M,TTvector{TA,N}}, want::AbstractVector{Int};
                         seed::Int=1234, orthogonal::Bool=true, timer::TimerOutput=TimerOutput()) where {TA,N,M}
  active = cache.groups[end]
  frozen_cols = zeros(Int, N+1)
  @inbounds for gi in 1:length(cache.groups)-1, l in 1:N+1
    frozen_cols[l] += cache.groups[gi].counts[l] * cache.groups[gi].brv[l]
  end
  target = copy(active.counts)
  @inbounds for l in 1:N
    need = max(0, want[l] - frozen_cols[l])
    target[l] = max(active.counts[l], cld(need, active.brv[l]))
  end
  @inbounds for l in 2:N
    target[l] = max(target[l], target[l-1])
  end
  extend_kronecker_sketch!(active, A, target; seed=seed, orthogonal=orthogonal, timer=timer)
  return cache
end

"""
    sketch_matrix(cache::KroneckerCachedSketch, l, prod_rks; weighting=:equal, nsamp=nothing, out=nothing) -> Matrix

Normalized 2-D Kronecker sketch at bond `l` (combines all groups via `finalize_cols_kronecker`).
"""
sketch_matrix(cache::KroneckerCachedSketch, l::Int, prod_rks::Int;
              weighting::Symbol=:equal, nsamp=nothing, out=nothing) =
  finalize_cols_kronecker(cache.groups, l, prod_rks; weighting=weighting, nsamp=nsamp, out=out)

"""
    remat_into!(Wj, l, cache::KroneckerCachedSketch, prod_rks, nsamp; weighting) -> Wj[l]

Materialize bond `l` of a Kronecker `cache` (prefix `nsamp`) into a **reused capacity buffer**
`Wj[l]` (a 2-D matrix), geometric-growing it only when the needed width exceeds capacity. Consumers
index columns within the materialized width (the spare capacity past it holds stale data).
"""
function remat_into!(Wj::Vector{Matrix{T}}, l::Int, cache::KroneckerCachedSketch, prod_rks::Int,
                     nsamp=nothing; weighting::Symbol=:equal) where {T}
  cnt = nsamp === nothing ? [g.counts[l] for g in cache.groups] : nsamp
  cols = sum(cnt[gi]*cache.groups[gi].brv[l] for gi in eachindex(cache.groups))
  if !isassigned(Wj, l) || size(Wj[l], 1) != prod_rks || size(Wj[l], 2) < cols
    newcap = (isassigned(Wj, l) && size(Wj[l], 1) == prod_rks) ? max(2*size(Wj[l], 2), cols) : cols
    Wj[l] = Matrix{T}(undef, prod_rks, newcap)
  end
  finalize_cols_kronecker(cache.groups, l, prod_rks; weighting=weighting, nsamp=nsamp, out=view(Wj[l], :, 1:cols))
  return Wj[l]
end
