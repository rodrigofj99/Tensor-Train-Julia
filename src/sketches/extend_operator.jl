"""
Incremental / extensible recursive **operator** sketches (reverse sweep): a sketch of `A·y` where
`A::TToperator` and `y::TTvector`. The operator analogue of `SketchGroup` — same global-sample
addressing, same raw (un-normalized) storage, same deficient-bonds-right-to-left extension — but the
per-bond partial carries the operator rank, so `W[l] :: (y.ttv_rks[l], A.tto_rks[l], brv[l],
counts[l])` (4-D) and the per-sample contraction is the 5-arg operator kernel
`contract_sketch_core_backwards!(W, y_core, A_core, S, V)` (no batched variant). Boundary
`W[N+1] = ones(1,1,1,1)`.

The random blocks are shared with the vector group at the same `(seed, bond, sample, reverse, group)`:
a uniform `block_rks` sketch of `A·y` and of `b` draw the **same** blocks, so `S(A·y) − S(b)` is the
sketch of the residual `A·y − b`. `1/√count` normalization is applied only in `finalize_cols_operator`.
"""
mutable struct OperatorSketchGroup{T}
  W::Vector{Array{T,4}}     # raw partials; W[l] :: (y.ttv_rks[l], A.tto_rks[l], brv[l], counts[l]); boundary = ones(1,1,1,1)
  counts::Vector{Int}       # per-bond sample count; boundary bond = 1 (tiled)
  brv::Vector{Int}          # block_rks_vec for this group's block_rks
  block_rks::Int
  group::Int                # seed namespace (disjoint draws across groups)
  reverse::Bool             # sweep direction: true = right→left (boundary N+1), false = left→right (boundary 1)
end

"""
    OperatorSketchGroup(T, A, y, block_rks, group; reverse=true) -> empty OperatorSketchGroup

Build an empty operator group (no samples yet) for the sketch of `A·y`, with sketch element type
derived from `T`, `eltype(A)`, and `eltype(y)`. `brv = block_rks_vec` is computed from `block_rks`
and `y`'s dimensions (identical to the vector group's, so the blocks line up). `reverse` selects the
sweep direction (right→left fills bonds 1..N reusing the right neighbour; left→right is the mirror).
"""
function OperatorSketchGroup(::Type{T}, A::TToperator{TA,N}, y::TTvector{Ty,N}, block_rks::Int, group::Int; reverse::Bool=true) where {T<:Number,TA<:Number,Ty<:Number,N}
  TW = typeof(one(T)*one(TA)*one(Ty))
  dims = y.ttv_dims
  brv = ones(Int, N+1)
  if reverse
    brv[1:N] .= block_rks
    for k = N:-1:1
      brv[k] = min(brv[k], dims[k]*brv[k+1])
    end
  else
    brv[2:N+1] .= block_rks
    for k = 1:N
      brv[k+1] = min(brv[k+1], dims[k]*brv[k])
    end
  end
  W = Vector{Array{TW,4}}(undef, N+1)
  counts = zeros(Int, N+1)
  bidx = reverse ? N+1 : 1
  W[bidx] = ones(TW, 1, 1, 1, 1); counts[bidx] = 1
  return OperatorSketchGroup{TW}(W, counts, brv, block_rks, group, reverse)
end

# Geometric-growth append of the new slabs (positions off+1..target_l) of `Wnew` into g.W[l]'s
# sample dimension (dim 4). Mirrors the vector group's _append_slabs! for 4-D operator partials.
function _append_slabs_op!(g::OperatorSketchGroup{TW}, l::Int, off::Int, target_l::Int, Wnew::Array{TW,4}) where {TW}
  if off == 0
    g.W[l] = Wnew
  else
    Wl = g.W[l]
    if size(Wl, 4) < target_l
      newcap = max(2*size(Wl, 4), target_l)
      buf = Array{TW,4}(undef, size(Wl, 1), size(Wl, 2), size(Wl, 3), newcap)
      copyto!(view(buf, :, :, :, 1:off), view(Wl, :, :, :, 1:off))
      Wl = buf; g.W[l] = buf
    end
    copyto!(view(Wl, :, :, :, off+1:target_l), Wnew)
  end
end

"""
    extend_operator_sketch!(g, A, y, target; seed, orthogonal, timer) -> g

Grow operator group `g` so each bond `l` has at least `target[l]` samples, computing only the missing
samples. For a reverse group `target` is non-decreasing over `1..N` (reuses the right neighbour); for a
forward group it is non-increasing over `2..N+1` (reuses the left neighbour).
"""
function extend_operator_sketch!(g::OperatorSketchGroup{TW}, A::TToperator{TA,N}, y::TTvector{Ty,N},
                                  target::AbstractVector{Int}; seed::Int=1234, orthogonal::Bool=true,
                                  timer::TimerOutput=TimerOutput()) where {TW,TA,Ty,N}
  dims = y.ttv_dims
  @timeit timer "extend_operator_sketch" begin
    if g.reverse
      # Size the per-call block / contraction buffers for the largest deficient bond, then reuse them
      # across bonds under GC.@preserve (as tt_recursive_sketch does).
      max_sketch = 0; cb1 = 0; cb2 = 0; cb3 = 0
      @inbounds for k = N:-1:1
        g.counts[k] >= target[k] && continue
        add = target[k] - g.counts[k]
        max_sketch = max(max_sketch, g.brv[k+1]*dims[k]*g.brv[k]*add)
        b1, b2, b3 = contract_sketch_core_backwards_operator_buffers_size(
                       dims[k], dims[k], y.ttv_rks[k], y.ttv_rks[k+1],
                       A.tto_rks[k], A.tto_rks[k+1], g.brv[k], g.brv[k+1])
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
          # Block shape (brv[k+1], dims[k], brv[k]) matches the kernel's S=(γ,ζ,c) — no permute.
          B = generate_sketch_blocks(seed, k, off, TW, g.brv[k+1], dims[k], g.brv[k], add, orthogonal;
                                     reverse=true, group=g.group, buffer=sketch_buffer, timer=timer)
          Wk_new = zeros(TW, y.ttv_rks[k], A.tto_rks[k], g.brv[k], add)
          # Recurse per sample against the right neighbour's existing column (tiled boundary for k=N).
          for s in 1:add
            Vs = k < N ? view(g.W[k+1], :, :, :, off+s) : view(g.W[N+1], :, :, :, 1)
            contract_sketch_core_backwards!(view(Wk_new, :, :, :, s), y.ttv_vec[k], A.tto_vec[k],
                                            view(B, :, :, :, s), Vs; buffer=contract_buffer)
          end
          _append_slabs_op!(g, k, off, target[k], Wk_new)
          g.counts[k] = target[k]
        end
      end
    else
      # Forward mirror: fill bonds l=k+1 (k=1..N) reusing the LEFT neighbour W[k]; boundary at bond 1.
      max_sketch = 0; cb1 = 0; cb2 = 0; cb3 = 0
      @inbounds for k = 1:N
        l = k + 1
        g.counts[l] >= target[l] && continue
        add = target[l] - g.counts[l]
        max_sketch = max(max_sketch, g.brv[k]*dims[k]*g.brv[k+1]*add)
        b1, b2, b3 = contract_sketch_core_forwards_operator_buffers_size(
                       dims[k], dims[k], y.ttv_rks[k], y.ttv_rks[k+1],
                       A.tto_rks[k], A.tto_rks[k+1], g.brv[k], g.brv[k+1])
        cb1 = max(cb1, b1); cb2 = max(cb2, b2); cb3 = max(cb3, b3)
      end
      max_sketch == 0 && return g
      sketch_buffer   = Vector{TW}(undef, max_sketch)
      contract_buffer = (Vector{TW}(undef, cb1), Vector{TW}(undef, cb2), Vector{TW}(undef, cb3))
      GC.@preserve sketch_buffer contract_buffer begin
        @inbounds for k = 1:N
          l = k + 1
          g.counts[l] >= target[l] && continue
          @assert k == 1 || target[l] <= g.counts[k] "non-increasing counts required for the forward recursion: bond $l target $(target[l]) exceeds left-neighbour count $(g.counts[k])"
          off = g.counts[l]; add = target[l] - off
          # Block shape (brv[k], dims[k], brv[k+1]) matches the forward kernel's S=(γ,z,c) — no permute.
          B = generate_sketch_blocks(seed, k, off, TW, g.brv[k], dims[k], g.brv[k+1], add, orthogonal;
                                     reverse=false, group=g.group, buffer=sketch_buffer, timer=timer)
          Wl_new = zeros(TW, y.ttv_rks[l], A.tto_rks[l], g.brv[l], add)
          for s in 1:add
            Vs = k > 1 ? view(g.W[k], :, :, :, off+s) : view(g.W[1], :, :, :, 1)
            contract_sketch_core_forwards!(view(Wl_new, :, :, :, s), y.ttv_vec[k], A.tto_vec[k],
                                           view(B, :, :, :, s), Vs; buffer=contract_buffer)
          end
          _append_slabs_op!(g, l, off, target[l], Wl_new)
          g.counts[l] = target[l]
        end
      end
    end
  end
  return g
end

# Per-group sample variance of the leading per-sample slab norms (operator slab is W[:,:,:,s]).
function _slab_var(g::OperatorSketchGroup, l::Int, p::Int)
  p <= 1 && return 1.0
  W = g.W[l]
  e = [sum(abs2, @view W[:, :, :, s]) for s in 1:p]
  m = sum(e)/p
  return sum(x->(x-m)^2, e)/(p-1)
end

"""
    finalize_cols_operator(groups, l, a_rks, op_rks; weighting=:equal, nsamp=nothing, out=nothing) -> Array{T,3}

Combine the operator groups' raw partials at bond `l` into the normalized sketch
`(a_rks, op_rks, total_cols)`, an unbiased isometry. `a_rks = y.ttv_rks[l]`, `op_rks = A.tto_rks[l]`.
The single-group `:equal` case is bit-identical to `tt_recursive_sketch(A, y)`'s reshaped `W[l]`.
"""
function finalize_cols_operator(groups::AbstractVector{<:OperatorSketchGroup{T}}, l::Int,
                                a_rks::Int, op_rks::Int;
                                weighting::Symbol=:equal, nsamp=nothing, out=nothing) where {T}
  cnt = nsamp === nothing ? [g.counts[l] for g in groups] : nsamp
  @assert all(cnt[gi] <= groups[gi].counts[l] for gi in eachindex(groups)) "finalize_cols_operator: nsamp exceeds available samples"
  ws = _group_weights(groups, l, cnt, weighting)
  total = sum(cnt[gi]*groups[gi].brv[l] for gi in eachindex(groups))
  dest = out === nothing ? Array{T,3}(undef, a_rks, op_rks, total) : out
  coff = 0
  for (gi, g) in enumerate(groups)
    cnt[gi] == 0 && continue
    bc = g.brv[l]*cnt[gi]
    # View the leading cnt[gi] slabs (g.W[l] may carry more samples and/or spare capacity);
    # reshape (a, op, brv, cnt) → (a, op, brv*cnt), column-major matching the reference reshape.
    M = reshape(view(g.W[l], :, :, :, 1:cnt[gi]), a_rks, op_rks, bc)
    @views dest[:, :, coff+1:coff+bc] .= M ./ sqrt(cnt[gi] / ws[gi])
    coff += bc
  end
  return dest
end

# ── Per-vector operator cache: one block-rks group (uniform) or two (mixed block_rks/block_rks_inc) ──
struct OperatorCachedSketch{T}
  groups::Vector{OperatorSketchGroup{T}}
end

"""
    cached_operator_sketch(T, A, y, block_rks, block_rks_inc, init_rank; reverse=true, seed, orthogonal, timer) -> OperatorCachedSketch

Build a reusable operator sketch cache for `A·y`: an initial (frozen) group of block rank `block_rks`
sized to the `init_rank` heuristic, plus — when `block_rks_inc != block_rks` — an empty extension
group of block rank `block_rks_inc`. `reverse` selects the sweep direction (all groups share it).
Grow it with `ensure_columns!`; read with `sketch_array`.
"""
function cached_operator_sketch(::Type{T}, A::TToperator{TA,N}, y::TTvector{Ty,N},
                                block_rks::Int, block_rks_inc::Int, init_rank::Int;
                                reverse::Bool=true, seed::Int=1234, orthogonal::Bool=true, timer::TimerOutput=TimerOutput()) where {T<:Number,TA<:Number,Ty<:Number,N}
  init = OperatorSketchGroup(T, A, y, block_rks, 0; reverse=reverse)
  extend_operator_sketch!(init, A, y, _heuristic_p(init_rank, init.brv, N; reverse=reverse); seed=seed, orthogonal=orthogonal, timer=timer)
  TW = eltype(init.W[reverse ? N+1 : 1])
  groups = block_rks_inc == block_rks ? OperatorSketchGroup{TW}[init] :
                                        OperatorSketchGroup{TW}[init, OperatorSketchGroup(T, A, y, block_rks_inc, 1; reverse=reverse)]
  return OperatorCachedSketch{TW}(groups)
end

"""
    ensure_columns!(cache::OperatorCachedSketch, A, y, want; seed, orthogonal, timer) -> cache

Grow the cache's active group so each bond `l` has at least `want[l]` *total* sketch columns
(across all groups). Frozen groups are untouched; only the missing samples of the active group are
computed. Mirrors the vector `ensure_columns!`.
"""
function ensure_columns!(cache::OperatorCachedSketch, A::TToperator{TA,N}, y::TTvector{Ty,N},
                         want::AbstractVector{Int};
                         seed::Int=1234, orthogonal::Bool=true, timer::TimerOutput=TimerOutput()) where {TA,Ty,N}
  active = cache.groups[end]
  frozen_cols = zeros(Int, N+1)
  @inbounds for gi in 1:length(cache.groups)-1, l in 1:N+1
    frozen_cols[l] += cache.groups[gi].counts[l] * cache.groups[gi].brv[l]
  end
  target = copy(active.counts)
  bonds = active.reverse ? (1:N) : (2:N+1)
  @inbounds for l in bonds
    need = max(0, want[l] - frozen_cols[l])
    target[l] = max(active.counts[l], cld(need, active.brv[l]))
  end
  if active.reverse
    @inbounds for l in 2:N
      target[l] = max(target[l], target[l-1])
    end
  else
    @inbounds for l in N:-1:2
      target[l] = max(target[l], target[l+1])
    end
  end
  extend_operator_sketch!(active, A, y, target; seed=seed, orthogonal=orthogonal, timer=timer)
  return cache
end

"""
    sketch_array(cache, l, a_rks, op_rks; weighting=:equal, nsamp=nothing, out=nothing) -> Array{T,3}

Normalized operator sketch at bond `l` (combines all groups via `finalize_cols_operator`).
"""
sketch_array(cache::OperatorCachedSketch, l::Int, a_rks::Int, op_rks::Int;
             weighting::Symbol=:equal, nsamp=nothing, out=nothing) =
  finalize_cols_operator(cache.groups, l, a_rks, op_rks; weighting=weighting, nsamp=nsamp, out=out)

"""
    remat_into_operator!(Wj, l, cache, a_rks, op_rks, nsamp; weighting) -> Wj[l]

Materialize bond `l` of an operator `cache` (prefix `nsamp`) into a **reused capacity buffer**
`Wj[l]` (a 3-D array), geometric-growing it only when the needed width exceeds capacity. Consumers
index columns within the materialized width (the spare capacity past it holds stale data).
"""
function remat_into_operator!(Wj::Vector{Array{T,3}}, l::Int, cache::OperatorCachedSketch,
                              a_rks::Int, op_rks::Int, nsamp=nothing; weighting::Symbol=:equal) where {T}
  cnt = nsamp === nothing ? [g.counts[l] for g in cache.groups] : nsamp
  cols = sum(cnt[gi]*cache.groups[gi].brv[l] for gi in eachindex(cache.groups))
  if !isassigned(Wj, l) || size(Wj[l], 1) != a_rks || size(Wj[l], 2) != op_rks || size(Wj[l], 3) < cols
    newcap = (isassigned(Wj, l) && size(Wj[l], 1) == a_rks && size(Wj[l], 2) == op_rks) ? max(2*size(Wj[l], 3), cols) : cols
    Wj[l] = Array{T,3}(undef, a_rks, op_rks, newcap)
  end
  finalize_cols_operator(cache.groups, l, a_rks, op_rks; weighting=weighting, nsamp=nsamp, out=view(Wj[l], :, :, 1:cols))
  return Wj[l]
end
