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
    @inbounds for k = N:-1:1
      g.counts[k] >= target[k] && continue
      # k<N recurses against the right neighbour's columns (needs them present); k=N recurses
      # against the tiled ones boundary, so target[N] is unconstrained.
      @assert k == N || target[k] <= g.counts[k+1] "non-decreasing counts required for the reverse recursion: bond $k target $(target[k]) exceeds right-neighbour count $(g.counts[k+1])"
      off = g.counts[k]; add = target[k] - off
      # Blocks at global sample indices off+1..off+add at bond k (group namespace g.group).
      Bβzbp = generate_sketch_blocks(seed, k, off, TW, g.brv[k+1], dims[k], g.brv[k], add, orthogonal;
                                     reverse=true, group=g.group, timer=timer)
      S = permutedims(Bβzbp, (2,1,3,4))           # (z, β, b, add) layout the kernel wants
      Wk_new = zeros(TW, A.ttv_rks[k], g.brv[k], add)
      # Recurse against the right neighbour's EXISTING columns at the same global positions
      # (or the tiled ones boundary for k = N) — this is what makes a sample globally addressable.
      V = k < N ? Array(view(g.W[k+1], :, :, off+1:target[k])) : repeat(g.W[N+1], 1, 1, add)
      contract_sketch_core_backwards!(Wk_new, A.ttv_vec[k], S, V)
      g.W[k] = off == 0 ? Wk_new : cat(g.W[k], Wk_new; dims=3)
      g.counts[k] = target[k]
    end
  end
  return g
end

# Per-group scalar weights w_g (Σ w_g = 1) at bond l for the chosen averaging scheme. Combining
# unbiased per-block estimators of unequal variance ⇒ inverse-variance (precision) weighting is the
# BLUE; :equal / :column are oblivious closed-form approximations (see PLAN.md / Stage 6).
function _group_weights(groups, l::Int, weighting::Symbol)
  if weighting === :equal
    raw = Float64[g.counts[l] for g in groups]                       # ∝ samples → overall 1/√Σcounts
  elseif weighting === :column
    raw = Float64[g.counts[l]*g.brv[l] for g in groups]              # ∝ columns (embedding-dim model)
  elseif weighting === :precision
    # Provisional empirical inverse-variance: per-group sample variance of raw per-sample slab norms
    # at this bond. Exact per-bond-residual form is finalized in Stage 6.
    raw = Float64[g.counts[l] == 0 ? 0.0 : g.counts[l] / max(_slab_var(g, l), eps()) for g in groups]
  else
    error("unknown block-averaging weighting :$weighting (use :equal, :column, or :precision)")
  end
  s = sum(raw)
  return raw ./ s
end

function _slab_var(g::SketchGroup, l::Int)
  p = g.counts[l]
  p <= 1 && return 1.0
  W = g.W[l]
  e = [sum(abs2, @view W[:, :, s]) for s in 1:p]
  m = sum(e)/p
  return sum(x->(x-m)^2, e)/(p-1)
end

"""
    finalize_cols(groups, l, rks_l; weighting=:equal) -> Matrix

Combine the groups' raw partials at bond `l` into the normalized sketch columns
`[√w₁ · group₁ | √w₂ · group₂ | …] · (per-group 1/√count)`, an unbiased isometry for any
`Σ w_g = 1`. `rks_l = A.ttv_rks[l]` (the row dimension).
"""
function finalize_cols(groups::AbstractVector{<:SketchGroup{T}}, l::Int, rks_l::Int;
                       weighting::Symbol=:equal) where {T}
  ws = _group_weights(groups, l, weighting)
  blocks = Matrix{T}[]
  for (gi, g) in enumerate(groups)
    g.counts[l] == 0 && continue
    M = reshape(g.W[l], rks_l, g.brv[l]*g.counts[l])
    # Divide (not multiply-by-reciprocal) so the single-group :equal case is bit-identical to
    # tt_recursive_sketch's `W ./= sqrt(count)`.
    push!(blocks, M ./ sqrt(g.counts[l] / ws[gi]))
  end
  return reduce(hcat, blocks)
end
