"""
    tt_combined_sketch([T=Float64,] A::TTvector, rks_or_rmax, s::Int; ...) -> (sketch, W, sketch_rks)
    tt_combined_sketch([T=Float64,] H::TToperator, A::TTvector, rks_or_rmax, s::Int; ...) -> (sketch, W, sketch_rks)
    tt_combined_sketch([T=Float64,] A::NTuple{M,TTvector}, rks_or_rmax, s::Int; ...) -> (sketch, W, sketch_rks)

Combined sketch returning both intermediate sketches (for `ttrand_rounding`) and an enlarged
boundary sketch of size ≥ s (for Rayleigh-Ritz or sketched projections).

The `rks_or_rmax` argument controls the intermediate sketch ranks exactly as in
`tt_recursive_sketch`. The additional `s` argument specifies the desired size of the
boundary sketch (W[1] for reverse=true, W[N+1] for reverse=false).

If `s ≤ sketch_rks[boundary]` (the boundary sketch already produced by `tt_recursive_sketch`),
the boundary sketch is returned as-is. Otherwise an additional `tt_sketch` call with `seed+1`
is appended (horizontally concatenated on the sketch dimension) to reach size ≥ s.

# Returns
- `sketch`: boundary sketch of last-dimension size s_actual ≥ s
- `W`: length N+1 array identical to `tt_recursive_sketch` output except
  `W[boundary] = ones([1])` (trivial placeholder; the real boundary sketch is `sketch`)
- `sketch_rks`: same as returned by `tt_recursive_sketch`
"""
function tt_combined_sketch(::Type{T}, A::TTvector{TA,N}, rks::Vector{Int}, s::Int;
                             reverse::Bool=true, orthogonal::Bool=true,
                             seed::Int=1234, block_rks::Int=N,
                             timer::TimerOutput=TimerOutput()) where {T<:Number, TA<:Number, N}

  boundary = reverse ? 1 : N+1

  rks = deepcopy(rks)
  rks[boundary] = max(rks[boundary], min(s, rks[reverse ? 2 : N]))
  W, sketch_rks = tt_recursive_sketch(T, A, rks; reverse=reverse, orthogonal=orthogonal,
                                       seed=seed, block_rks=block_rks, timer=timer)
  sketch = vec(W[boundary])

  TW = eltype(W[1])
  if s > length(sketch)
    extra, _ = tt_sketch(T, A, s - length(sketch); reverse=reverse, orthogonal=orthogonal,
                            seed=seed+1, block_rks=block_rks)
    s = length(sketch) + length(extra)
    sketch = [sqrt(length(sketch)/s) .* sketch; sqrt(length(extra)/s) .* extra]
  end
  W[boundary] = ones(TW, 1, 1)
  sketch_rks[boundary] = 1
  return sketch, W, sketch_rks
end

function tt_combined_sketch(::Type{T}, A::TTvector{TA,N}, rmax::Int, s::Int;
                             reverse::Bool=true, kwargs...) where {T<:Number, TA<:Number, N}
  rks = rmax * ones(Int, N+1)
  rks[(reverse ? N+1 : 1)] = 1
  return tt_combined_sketch(T, A, rks, s; reverse=reverse, kwargs...)
end

function tt_combined_sketch(A::TTvector{T,N}, rks_or_rmax, s::Int; kwargs...) where {T<:Number, N}
  return tt_combined_sketch(Float64, A, rks_or_rmax, s; kwargs...)
end

function tt_combined_sketch(::Type{T}, H::TToperator{TH,N}, A::TTvector{TA,N}, rks::Vector{Int}, s::Int;
                             reverse::Bool=true, orthogonal::Bool=true,
                             seed::Int=1234, block_rks::Int=N,
                             timer::TimerOutput=TimerOutput()) where {T<:Number, TH<:Number, TA<:Number, N}
  boundary = reverse ? 1 : N+1

  rks = deepcopy(rks)
  rks[boundary] = max(rks[boundary], min(s, rks[reverse ? 2 : N]))

  W, sketch_rks = tt_recursive_sketch(T, H, A, rks; reverse=reverse, orthogonal=orthogonal,
                                       seed=seed, block_rks=block_rks, timer=timer)
  sketch = vec(W[boundary])

  TW = eltype(W[1])
  s_rec = sketch_rks[boundary]
  if s > s_rec
    extra, _ = tt_sketch(T, H, A, s - s_rec; reverse=reverse, orthogonal=orthogonal,
                            seed=seed+1, block_rks=block_rks)
    s = s_rec + length(extra)
    sketch = [sqrt(length(sketch)/s) .* sketch; sqrt(length(extra)/s) .* extra]
  end
  W[boundary] = ones(TW, 1, 1, 1)
  sketch_rks[boundary] = 1
  return sketch, W, sketch_rks
end

function tt_combined_sketch(::Type{T}, H::TToperator{TH,N}, A::TTvector{TA,N}, rmax::Int, s::Int;
                             reverse::Bool=true, kwargs...) where {T<:Number, TH<:Number, TA<:Number, N}
  rks = rmax * ones(Int, N+1)
  rks[(reverse ? N+1 : 1)] = 1
  return tt_combined_sketch(T, H, A, rks, s; reverse=reverse, kwargs...)
end

function tt_combined_sketch(H::TToperator{TH,N}, A::TTvector{TA,N}, rks_or_rmax, s::Int; kwargs...) where {TH<:Number, TA<:Number, N}
  return tt_combined_sketch(Float64, H, A, rks_or_rmax, s; kwargs...)
end

function tt_combined_sketch(::Type{T}, A::NTuple{M,TTvector{TA,N}}, rks, s::Int;
                             reverse::Bool=true, orthogonal::Bool=true,
                             seed::Int=1234, block_rks::Int=N,
                             timer::TimerOutput=TimerOutput()) where {T<:Number, TA<:Number, N, M}
  boundary = reverse ? 1 : N+1

  rks = deepcopy(rks)
  rks[boundary] = max(rks[boundary], min(s, rks[reverse ? 2 : N]))

  W, sketch_rks = tt_recursive_sketch(T, A, rks; reverse=reverse, orthogonal=orthogonal,
                                       seed=seed, block_rks=block_rks, timer=timer)
  sketch = vec(W[boundary])

  TW = eltype(W[1])
  s_rec = sketch_rks[boundary]
  if s > s_rec
    extra, _ = tt_sketch(T, A, s - s_rec; reverse=reverse, orthogonal=orthogonal,
                            seed=seed+1, block_rks=block_rks)
    s = s_rec + length(extra)
    sketch = [sqrt(length(sketch)/s) .* sketch; sqrt(length(extra)/s) .* extra]
  end
  W[boundary] = ones(TW, ntuple(i->1, M+1)...)
  sketch_rks[boundary] = 1
  return sketch, W, sketch_rks
end

function tt_combined_sketch(::Type{T}, A::NTuple{M,TTvector{TA,N}}, rmax::Int, s::Int;
                             reverse::Bool=true, kwargs...) where {T<:Number, TA<:Number, N, M}
  rks = rmax * ones(Int, N+1)
  rks[(reverse ? N+1 : 1)] = 1
  return tt_combined_sketch(T, A, rks, s; reverse=reverse, kwargs...)
end

function tt_combined_sketch(A::NTuple{M,TTvector{T,N}}, rks_or_rmax, s::Int; kwargs...) where {T<:Number, N, M}
  return tt_combined_sketch(Float64, A, rks_or_rmax, s; kwargs...)
end
