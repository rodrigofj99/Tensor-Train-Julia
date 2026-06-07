"""
    tt_sketch([T=Float64,] A::TTvector, s::Int; reverse=true, orthogonal=true, seed=1234, block_rks=N) -> (W, s_actual)

Compute only the boundary sketch without storing intermediate contractions.

Returns the single boundary sketch matrix:
- `W[1]`   of shape `(1, s_actual)` when `reverse=true`  (right-to-left sweep)
- `W[N+1]` of shape `(1, s_actual)` when `reverse=false` (left-to-right sweep)

`s` is the desired sketch size; `s_actual = b_rks * p` may be slightly larger than `s` where
`b_rks` is the effective block rank at the boundary and `p = cld(s, b_rks)`.

Each of the `p` sketch blocks is propagated independently through the full chain (outer loop
over `p`, inner loop over sites), so at any point only a single 2D state matrix `w` is live
in addition to the pre-allocated sketch and contraction buffers. The `p` boundary slices are
written directly into the result matrix as they are produced.

# Arguments
- `T::Type`: Element type for the sketch tensor (default: Float64)
- `A::TTvector{TA,N}`: Input TTvector
- `s::Int`: Desired sketch size at the boundary

# Keyword Arguments
- `reverse::Bool=true`: Sweep direction (true: right-to-left, false: left-to-right)
- `orthogonal::Bool=true`: Use QR-orthogonal random sketch blocks
- `seed::Int=1234`: Random seed
- `block_rks::Int=N`: Block rank controlling sketch granularity
"""
function tt_sketch(::Type{T}, A::TTvector{TA,N}, s::Int;
                         reverse::Bool=true, orthogonal::Bool=true,
                         seed::Int=1234, block_rks::Int=N) where {T<:Number, TA<:Number, N}

  dims = A.ttv_dims
  TW = typeof(one(T) * one(TA))

  # Block rank vector (capped to respect physical dimensions)
  block_rks_vec = ones(Int, N+1)
  if reverse
    block_rks_vec[1:N] .= block_rks
    for k = N:-1:1
      block_rks_vec[k] = min(block_rks_vec[k], dims[k] * block_rks_vec[k+1])
    end
  else
    block_rks_vec[2:N+1] .= block_rks
    for k = 1:N
      block_rks_vec[k+1] = min(block_rks_vec[k+1], dims[k] * block_rks_vec[k])
    end
  end

  # Constant p determined by boundary block rank
  b_rks = reverse ? block_rks_vec[1] : block_rks_vec[N+1]
  p = cld(s, b_rks)
  s_actual = b_rks * p

  # Preallocate sketch buffer (p blocks at a time, matching tt_recursive_sketch order)
  max_sketch_buffer_size = if reverse
    maximum(block_rks_vec[k+1] * dims[k] * block_rks_vec[k] * p for k in 1:N)
  else
    maximum(block_rks_vec[k] * dims[k] * block_rks_vec[k+1] * p for k in 1:N)
  end
  sketch_buffer = Vector{T}(undef, max_sketch_buffer_size)

  # Preallocate contraction buffers
  max_buf1 = 0
  max_buf2 = 0
  for k in 1:N
    z = dims[k]
    if reverse
      a, α, b, β = A.ttv_rks[k], A.ttv_rks[k+1], block_rks_vec[k], block_rks_vec[k+1]
      buf1, buf2 = contract_sketch_core_backwards_buffers_size(z, a, α, β, b)
    else
      α, a, β, b = A.ttv_rks[k], A.ttv_rks[k+1], block_rks_vec[k], block_rks_vec[k+1]
      buf1, buf2 = contract_sketch_core_forwards_buffers_size(z, α, a, β, b)
    end
    max_buf1 = max(max_buf1, buf1)
    max_buf2 = max(max_buf2, buf2)
  end
  contract_buffer = (Vector{TW}(undef, max_buf1), Vector{TW}(undef, max_buf2))

  if reverse
    W_prev = ones(TW, 1, 1, 1)  # W[N+1]: p-slice dimension is 1 (boundary)
    W_curr = W_prev
    @inbounds for k in N:-1:1
      B_sketch = generate_sketch_blocks(seed, k, 0, T, block_rks_vec[k+1], dims[k], block_rks_vec[k], p, orthogonal; reverse=true, buffer=sketch_buffer)
      W_curr = zeros(TW, A.ttv_rks[k], block_rks_vec[k], p)
      for j in 1:p
        contract_sketch_core_backwards!(view(W_curr,:,:,j), A.ttv_vec[k], view(B_sketch,:,:,:,j), view(W_prev,:,:,(k<N ? j : 1)); buffer=contract_buffer)
      end
      W_prev = W_curr
    end
  else
    W_prev = ones(TW, 1, 1, 1)  # W[1]: p-slice dimension is 1 (boundary)
    W_curr = W_prev
    @inbounds for k in 1:N
      B_sketch = generate_sketch_blocks(seed, k, 0, T, block_rks_vec[k], dims[k], block_rks_vec[k+1], p, orthogonal; reverse=false, buffer=sketch_buffer)
      W_curr = zeros(TW, A.ttv_rks[k+1], block_rks_vec[k+1], p)
      for j in 1:p
        contract_sketch_core_forwards!(view(W_curr,:,:,j), A.ttv_vec[k], view(B_sketch,:,:,:,j), view(W_prev,:,:,(k>1 ? j : 1)); buffer=contract_buffer)
      end
      W_prev = W_curr
    end
  end
  W_curr ./= sqrt(p)
  return vec(W_curr), s_actual
end

function tt_sketch(A::TTvector{T,N}, s::Int; kwargs...) where {T,N}
  return tt_sketch(Float64, A, s; kwargs...)
end

function tt_sketch(::Type{T}, H::TToperator{TH,N}, A::TTvector{TA,N}, s::Int;
                   reverse::Bool=true, orthogonal::Bool=true,
                   seed::Int=1234, block_rks::Int=N) where {T<:Number, TH<:Number, TA<:Number, N}

  dims = A.ttv_dims
  TW = typeof(one(T) * one(TA) * one(TH))

  # Block rank vector (capped to respect physical dimensions)
  block_rks_vec = ones(Int, N+1)
  if reverse
    block_rks_vec[1:N] .= block_rks
    for k = N:-1:1
      block_rks_vec[k] = min(block_rks_vec[k], dims[k] * block_rks_vec[k+1])
    end
  else
    block_rks_vec[2:N+1] .= block_rks
    for k = 1:N
      block_rks_vec[k+1] = min(block_rks_vec[k+1], dims[k] * block_rks_vec[k])
    end
  end

  # Constant p determined by boundary block rank
  b_rks = reverse ? block_rks_vec[1] : block_rks_vec[N+1]
  p = cld(s, b_rks)
  s_actual = b_rks * p

  # Preallocate sketch buffer (p blocks at a time, matching tt_recursive_sketch order)
  max_sketch_buffer_size = if reverse
    maximum(block_rks_vec[k+1] * dims[k] * block_rks_vec[k] * p for k in 1:N)
  else
    maximum(block_rks_vec[k] * dims[k] * block_rks_vec[k+1] * p for k in 1:N)
  end
  sketch_buffer = Vector{T}(undef, max_sketch_buffer_size)

  # Preallocate contraction buffers (3 for the operator case)
  max_buf1 = 0
  max_buf2 = 0
  max_buf3 = 0
  for k in 1:N
    z = dims[k]
    if reverse
      a, α = A.ttv_rks[k], A.ttv_rks[k+1]
      b, β = H.tto_rks[k], H.tto_rks[k+1]
      c, γ = block_rks_vec[k], block_rks_vec[k+1]
      buf1, buf2, buf3 = contract_sketch_core_backwards_operator_buffers_size(z, z, a, α, b, β, c, γ)
    else
      α, a = A.ttv_rks[k], A.ttv_rks[k+1]
      β, b = H.tto_rks[k], H.tto_rks[k+1]
      γ, c = block_rks_vec[k], block_rks_vec[k+1]
      buf1, buf2, buf3 = contract_sketch_core_forwards_operator_buffers_size(z, z, α, a, β, b, γ, c)
    end
    max_buf1 = max(max_buf1, buf1)
    max_buf2 = max(max_buf2, buf2)
    max_buf3 = max(max_buf3, buf3)
  end
  contract_buffer = (Vector{TW}(undef, max_buf1), Vector{TW}(undef, max_buf2), Vector{TW}(undef, max_buf3))

  if reverse
    W_prev = ones(TW, 1, 1, 1, 1)  # W[N+1]: shape (rks_A, rks_H, b_rks, p), all 1 at boundary
    W_curr = W_prev
    @inbounds for k in N:-1:1
      B_sketch = generate_sketch_blocks(seed, k, 0, T, block_rks_vec[k+1], dims[k], block_rks_vec[k], p, orthogonal; reverse=true, buffer=sketch_buffer)
      W_curr = zeros(TW, A.ttv_rks[k], H.tto_rks[k], block_rks_vec[k], p)
      for j in 1:p
        contract_sketch_core_backwards!(view(W_curr,:,:,:,j), A.ttv_vec[k], H.tto_vec[k], view(B_sketch,:,:,:,j), view(W_prev,:,:,:,(k<N ? j : 1)); buffer=contract_buffer)
      end
      W_prev = W_curr
    end
  else
    W_prev = ones(TW, 1, 1, 1, 1)  # W[1]: shape (rks_A, rks_H, b_rks, p), all 1 at boundary
    W_curr = W_prev
    @inbounds for k in 1:N
      B_sketch = generate_sketch_blocks(seed, k, 0, T, block_rks_vec[k], dims[k], block_rks_vec[k+1], p, orthogonal; reverse=false, buffer=sketch_buffer)
      W_curr = zeros(TW, A.ttv_rks[k+1], H.tto_rks[k+1], block_rks_vec[k+1], p)
      for j in 1:p
        contract_sketch_core_forwards!(view(W_curr,:,:,:,j), A.ttv_vec[k], H.tto_vec[k], view(B_sketch,:,:,:,j), view(W_prev,:,:,:,(k>1 ? j : 1)); buffer=contract_buffer)
      end
      W_prev = W_curr
    end
  end
  W_curr ./= sqrt(p)
  return vec(W_curr), s_actual
end

function tt_sketch(H::TToperator{TH,N}, A::TTvector{TA,N}, s::Int; kwargs...) where {TH,TA,N}
  return tt_sketch(Float64, H, A, s; kwargs...)
end

function tt_sketch(::Type{T}, A::NTuple{M, TTvector{TA,N}}, s::Int;
                   reverse::Bool=true, orthogonal::Bool=true,
                   seed::Int=1234, block_rks::Int=N) where {T<:Number, TA<:Number, N, M}

  dims = A[1].ttv_dims
  TW = typeof(one(T) * one(TA))

  # Block rank vector (capped to respect physical dimensions)
  block_rks_vec = ones(Int, N+1)
  if reverse
    block_rks_vec[1:N] .= block_rks
    for k = N:-1:1
      block_rks_vec[k] = min(block_rks_vec[k], dims[k] * block_rks_vec[k+1])
    end
  else
    block_rks_vec[2:N+1] .= block_rks
    for k = 1:N
      block_rks_vec[k+1] = min(block_rks_vec[k+1], dims[k] * block_rks_vec[k])
    end
  end

  # Constant p determined by boundary block rank
  b_rks = reverse ? block_rks_vec[1] : block_rks_vec[N+1]
  p = cld(s, b_rks)
  s_actual = b_rks * p

  # Preallocate sketch buffer (p blocks at a time, matching tt_recursive_sketch order)
  max_sketch_buffer_size = if reverse
    maximum(block_rks_vec[k+1] * dims[k] * block_rks_vec[k] * p for k in 1:N)
  else
    maximum(block_rks_vec[k] * dims[k] * block_rks_vec[k+1] * p for k in 1:N)
  end
  sketch_buffer = Vector{T}(undef, max_sketch_buffer_size)

  # Preallocate contraction buffers
  max_buf1 = 0
  max_buf2 = 0
  max_buf3 = 0
  for k in 1:N
    z = dims[k]
    if reverse
      v = ntuple(i -> (i == M+1 ? block_rks_vec[k+1] : A[i].ttv_rks[k+1]), M+1)
      w = ntuple(i -> (i == M+1 ? block_rks_vec[k  ] : A[i].ttv_rks[k  ]), M+1)
      buf1, buf2, buf3 = contract_sketch_core_backwards_kronecker_buffers_size(z, v, w)
    else
      v = ntuple(i -> (i == M+1 ? block_rks_vec[k  ] : A[i].ttv_rks[k  ]), M+1)
      w = ntuple(i -> (i == M+1 ? block_rks_vec[k+1] : A[i].ttv_rks[k+1]), M+1)
      buf1, buf2, buf3 = contract_sketch_core_forwards_kronecker_buffers_size(z, v, w)
    end
    max_buf1 = max(max_buf1, buf1)
    max_buf2 = max(max_buf2, buf2)
    max_buf3 = max(max_buf3, buf3)
  end
  contract_buffer = (Vector{TW}(undef, max_buf1), Vector{TW}(undef, max_buf2), Vector{TW}(undef, max_buf3))

  if reverse
    W_prev = ones(TW, ntuple(i->1, M+2)...)  # W[N+1]: (M TT-rks, b_rks, p), all 1 at boundary
    W_curr = W_prev
    @inbounds for k in N:-1:1
      B_sketch = generate_sketch_blocks(seed, k, 0, T, block_rks_vec[k+1], dims[k], block_rks_vec[k], p, orthogonal; reverse=true, buffer=sketch_buffer)
      W_curr = zeros(TW, ntuple(i -> (i == M+1 ? block_rks_vec[k] : A[i].ttv_rks[k]), M+1)..., p)
      for j in 1:p
        contract_sketch_core_kronecker_backwards!(view(W_curr, ntuple(i->Colon(), M+1)..., j), ntuple(i -> A[i].ttv_vec[k], M), view(B_sketch,:,:,:,j), view(W_prev, ntuple(i->Colon(), M+1)..., (k<N ? j : 1)); buffer=contract_buffer)
      end
      W_prev = W_curr
    end
  else
    W_prev = ones(TW, ntuple(i->1, M+2)...)  # W[1]: (M TT-rks, b_rks, p), all 1 at boundary
    W_curr = W_prev
    @inbounds for k in 1:N
      B_sketch = generate_sketch_blocks(seed, k, 0, T, block_rks_vec[k], dims[k], block_rks_vec[k+1], p, orthogonal; reverse=false, buffer=sketch_buffer)
      W_curr = zeros(TW, ntuple(i -> (i == M+1 ? block_rks_vec[k+1] : A[i].ttv_rks[k+1]), M+1)..., p)
      for j in 1:p
        contract_sketch_core_kronecker_forwards!(view(W_curr, ntuple(i->Colon(), M+1)..., j), ntuple(i -> A[i].ttv_vec[k], M), view(B_sketch,:,:,:,j), view(W_prev, ntuple(i->Colon(), M+1)..., (k>1 ? j : 1)); buffer=contract_buffer)
      end
      W_prev = W_curr
    end
  end
  W_curr ./= sqrt(p)
  return vec(W_curr), s_actual
end

function tt_sketch(A::NTuple{M, TTvector{T,N}}, s::Int; kwargs...) where {T, N, M}
  return tt_sketch(Float64, A, s; kwargs...)
end
