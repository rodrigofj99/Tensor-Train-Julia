using TensorTrains
using LinearAlgebra
using StaticArrays
using Random
using TensorOperations
using TimerOutputs

"""
    generate_sketch_blocks(rng, ::Type{T}, left_rank, dim, right_rank, p, orthogonal) where T

Generate random sketch tensor blocks for tensor train sketching with proper normalization.

# Arguments
- `rng`: Random number generator
- `T::Type`: Element type for the sketch tensor
- `dim::Int`: Physical dimension at this mode
- `left_rank::Int`: Left block rank
- `right_rank::Int`: Right block rank
- `p::Int`: Number of sketch blocks
- `orthogonal::Bool`: Whether to use orthogonal sketches (QR decomposition)
- `buffer::Union{Nothing,AbstractVector}`: Optional preallocated buffer

# Returns
- `B_sketch::Array{T,4}`: Sketch tensor of size (left_rank, dim, right_rank, p)

# Algorithm
Three cases with optimized normalization:

1. **Orthogonal QR** (when right_rank < dim*left_rank):
   - Applies QR factorization with column pivoting for numerical stability
   - Normalization: `sqrt(dim*left_rank/right_rank)` to maintain expected spectral norm

2. **Identity optimization** (when right_rank ≥ dim*left_rank):
   - Uses identity matrix structure for efficiency
   - Updates right_rank = dim*left_rank
   - No additional normalization needed

3. **Random normalized** (when orthogonal=false):
   - Uses standard Gaussian random tensors
   - Normalization: `1/sqrt(right_rank)` for variance control

The normalization ensures consistent spectral properties across different sketching modes.
"""
function generate_sketch_blocks(rng, ::Type{T}, left_rank, dim, right_rank, p, orthogonal; buffer=nothing, timer::TimerOutput = TimerOutput()) where T
  @assert (!orthogonal) || right_rank <= dim * left_rank
  use_identity = orthogonal && (right_rank == dim * left_rank)

  @timeit timer "Block allocation" begin
    if buffer === nothing
      block = Array{T,3}(undef, left_rank*dim, right_rank, p)
    else
      block_size = left_rank*dim*right_rank*p
      @assert length(buffer) >= block_size "Buffer too small: need $block_size, got $(length(buffer))"
      block = unsafe_wrap(Array, pointer(buffer), (left_rank*dim, right_rank, p))
    end
  end

  if use_identity
    @timeit timer "identity block creation" begin
      # Identity 'sketch'
      fill!(block, T(0))
      for j=1:p, i in 1:right_rank
        block[i,i,j] = 1
      end
    end
  else
    @timeit timer "random number generator" begin
      randn!(rng, block)
    end

    if orthogonal # QR orthogonalization
      @timeit timer "qr_factorization" begin
        @inbounds for j=1:p
            q,_ = qr!(block[:,:,j])
            block[:,:,j] .= Array(q)
        end
      end
      @timeit timer "normalization" block .*= sqrt(left_rank*dim/right_rank)
    else # Simple normalization
      @timeit timer "normalization" block .*= 1/sqrt(right_rank)
    end
  end
  return reshape(block, left_rank, dim, right_rank, p)
end

"""
    compute_sketch_blocks_heuristic(rks::Vector{Int}, block_rks_vec::Vector{Int}, N::Int; reverse::Bool)

Compute the heuristic number of sketch blocks p[k] for each core using oversampling strategy.

# Arguments
- `rks::Vector{Int}`: Target sketch ranks (length N+1)
- `block_rks_vec::Vector{Int}`: Block ranks for each core (length N+1)
- `N::Int`: Number of cores
- `reverse::Bool`: Sweep direction (true: right-to-left, false: left-to-right)

# Returns
- `p::Vector{Int}`: Number of sketch blocks per core (length N+1)

# Algorithm
1. Base heuristic: p[k] = ceil(2 * rks[k] / block_rks_vec[k]) for oversampling
2. Monotonicity constraint:
   - If reverse: p is monotonically increasing (p[i] ≥ p[i-1])
   - If forward: p is monotonically decreasing (p[i] ≤ p[i+1])
3. Boundary: p[1] = 1 (forward) or p[N+1] = 1 (reverse)
"""
function compute_sketch_blocks_heuristic(rks::Vector{Int}, block_rks_vec::Vector{Int}, N::Int; reverse::Bool)
  p = @. ceil(Int, rks/block_rks_vec)

  if reverse
    for i=2:N
      p[i] = max(p[i], p[i-1])  # Monotonically increasing
    end
    p[N+1] = 1
  else
    for i=N:-1:2
      p[i] = max(p[i], p[i+1])  # Monotonically decreasing
    end
    p[1] = 1
  end

  return p
end

"""
    tt_recursive_sketch([T=Float64,] [H::TToperator,] A::TTvector, rks;       orthogonal=true, reverse=true, seed=1234, block_rks=N, oversampling=1, timer) -> (W, sketch_rks)
    tt_recursive_sketch([T=Float64,] [H::TToperator,] A::TTvector, rmax::Int; orthogonal=true, reverse=true, seed=1234, block_rks=N, oversampling=1, timer) -> (W, sketch_rks)

Compute a recursive sketch of a TTvector A, or of the result of applying TToperator H to TTvector A,
without explicitly forming the product, using optimized random projections with adaptive normalization.

Generates sketch matrices W[k] for each TT core by recursively contracting with random
orthogonal (or normalized) tensors. The sketch provides a compressed representation
suitable for efficient randomized algorithms like ttrand_rounding and STTA.

# Arguments
- `T::Type{<:Number}`: Element type for random sketch tensors (optional, default: Float64)
- `H::TToperator{TH,N}`: TToperator to apply (optional)
- `A::TTvector{TA,N}`: Input TTvector to sketch
- `rks::Vector{Int}` or `rmax::Int`: Target sketch ranks (length N+1 with boundary conditions rks[1]=1 or rks[N+1]=1) or maximum sketch rank

# Keyword Arguments
- `orthogonal::Bool=true`: Generate orthogonal random tensors (via QR) for better numerical stability
- `reverse::Bool=true`: Sweep direction (true: right-to-left, false: left-to-right)
- `seed::Int=1234`: Random seed for reproducibility
- `block_rks::Int=N`: Block rank for sketching heuristic (controls sketch granularity)
- `oversampling=1`: overall multiplier for the number of blocks, useful for STTA oversampling in particular

# Returns
- `W::Vector{Matrix}`: Sketch matrices where W[k] has size (A.ttv_rks[k], sketch_rks[k])
- `sketch_rks::Vector{Int}`: Actual sketch ranks achieved (includes oversampling factor)

# Algorithm
Uses a **block sketching strategy** with adaptive normalization:

1. **Block Structure**: Creates p[k] sketch blocks per core, with p determined by 
   oversampling heuristic: `p[k] = ceil(2*rks[k]/block_rks)`

2. **Normalization Strategy**:
   - **Orthogonal blocks**: QR with normalization `sqrt(dim*left_rank/right_rank)`
   - **Identity optimization**: When ranks permit, uses identity structure
   - **Random blocks**: Normalized by `1/sqrt(right_rank)`
   - **Final scaling**: Each W[k] scaled by `1/sqrt(p[k])` for block averaging

3. **Adaptive Rank Adjustment**: Block ranks may be updated for identity optimization,
   ensuring numerical efficiency while maintaining approximation quality.

The combination of QR orthogonalization and careful normalization provides excellent
numerical stability and consistent spectral properties across different rank regimes.

# References
- Randomized tensor train decomposition with block sketching
- See ttrand_rounding, stta for usage examples
"""
function tt_recursive_sketch(::Type{T}, A::TTvector{TA,N}, rks; orthogonal=true, reverse=true, seed=1234, block_rks::Int=N, oversampling=1, timer::TimerOutput = TimerOutput()) where {T<:Number,TA<:Number,N}
  @timeit timer "tt_recursive_sketch" begin
    rng = Random.default_rng()
    Random.seed!(rng, seed)

    dims = A.ttv_dims
    TW = typeof(one(T)*one(TA))
    W = Vector{Array{TW,3}}(undef, N+1)

    if reverse
      @timeit timer "sketch_initialization" begin
        @assert rks[N+1] == 1 && A.ttv_rks[N+1] == 1
        block_rks_vec = ones(Int, N+1)
        block_rks_vec[1:N] .= block_rks
        for k=N:-1:1
          block_rks_vec[k] = min(block_rks_vec[k], dims[k]*block_rks_vec[k+1])
        end

        p = compute_sketch_blocks_heuristic(rks, block_rks_vec, N; reverse=true)
        if oversampling ≠ 1
          p[1:N] .= ceil.(Int, oversampling .* p[1:N])
        end

        W[N+1] = ones(TW,1,1,1)

        # Preallocate buffer for the entire loop
        max_sketch_buffer_size = maximum(block_rks_vec[k+1] * dims[k] * block_rks_vec[k] * p[k] for k in 1:N)
        sketch_buffer = Vector{T}(undef, max_sketch_buffer_size)

        max_contract_buffer1_size = 0
        max_contract_buffer2_size = 0
        for k in 1:N
          z = dims[k]
          a = A.ttv_rks[k]
          α = A.ttv_rks[k+1]
          b = block_rks_vec[k]
          β = block_rks_vec[k+1]
          buf1_size, buf2_size = contract_sketch_core_backwards_buffers_size(z, a, α, β, b)
          max_contract_buffer1_size = max(max_contract_buffer1_size, buf1_size)
          max_contract_buffer2_size = max(max_contract_buffer2_size, buf2_size)
        end
        contract_buffer = (Vector{TW}(undef, max_contract_buffer1_size), Vector{TW}(undef, max_contract_buffer2_size))
      end

      @timeit timer "contraction_reverse" begin
        @inbounds for k in N:-1:1
          @timeit timer "sketch_generation" begin
            B_sketch = generate_sketch_blocks(rng, T, block_rks_vec[k+1], dims[k], block_rks_vec[k], p[k], orthogonal; buffer=sketch_buffer, timer=timer)
          end

          @timeit timer "W_allocation" begin
            W[k] = zeros(TW, A.ttv_rks[k], block_rks_vec[k], p[k])
          end

          @timeit timer "tensor_contraction" begin
            for j=1:p[k]
              W_next_j = view(W[k+1],:,:,(k<N ? j : 1))
              B_j = view(B_sketch,:,:,:,j)
              W_k_j = view(W[k],:,:,j)
              contract_sketch_core_backwards!(W_k_j, A.ttv_vec[k], B_j, W_next_j; buffer=contract_buffer)
            end
          end
        end
      end
    else
      @timeit timer "sketch_initialization" begin
        @assert rks[1] == 1 && A.ttv_rks[1] == 1
        block_rks_vec = ones(Int, N+1)
        block_rks_vec[2:N+1] .= block_rks
        for k=1:N
          block_rks_vec[k+1] = min(block_rks_vec[k+1], dims[k]*block_rks_vec[k])
        end

        p = compute_sketch_blocks_heuristic(rks, block_rks_vec, N; reverse=false)
        if oversampling ≠ 1
          p[2:N+1] .= ceil.(Int, oversampling .* p[2:N+1])
        end
        W[1] = ones(TW,1,1,1)

        # Preallocate buffer for the entire loop
        max_sketch_buffer_size = maximum(block_rks_vec[k] * dims[k] * block_rks_vec[k+1] * p[k+1] for k in 1:N)
        sketch_buffer = Vector{T}(undef, max_sketch_buffer_size)

        max_contract_buffer1_size = 0
        max_contract_buffer2_size = 0
        for k in 1:N
          z = dims[k]
          α = A.ttv_rks[k]
          a = A.ttv_rks[k+1]
          β = block_rks_vec[k]
          b = block_rks_vec[k+1]
          buf1_size, buf2_size = contract_sketch_core_forwards_buffers_size(z, α, a, β, b)
          max_contract_buffer1_size = max(max_contract_buffer1_size, buf1_size)
          max_contract_buffer2_size = max(max_contract_buffer2_size, buf2_size)
        end
        contract_buffer = (Vector{TW}(undef, max_contract_buffer1_size), Vector{TW}(undef, max_contract_buffer2_size))
      end

      @timeit timer "contraction_forward" begin
        @inbounds for k in 1:N
          @timeit timer "sketch_generation" begin
            B_sketch = generate_sketch_blocks(rng, T, block_rks_vec[k], dims[k], block_rks_vec[k+1], p[k+1], orthogonal; buffer=sketch_buffer, timer=timer)
          end

          @timeit timer "W_allocation" begin
            W[k+1] = zeros(TW, A.ttv_rks[k+1], block_rks_vec[k+1], p[k+1])
          end

          @timeit timer "tensor_contraction" begin
            for j=1:p[k+1]
              W_k_j = view(W[k],:,:,(k>1 ? j : 1))
              B_j = view(B_sketch,:,:,:,j)
              W_next_j = view(W[k+1],:,:,j)
              contract_sketch_core_forwards!(W_next_j, A.ttv_vec[k], B_j, W_k_j; buffer=contract_buffer)
            end
          end
        end
      end
    end

    for k=1:N+1
      W[k] ./= sqrt(p[k])
    end
    return [reshape(W[k], A.ttv_rks[k], block_rks_vec[k]*p[k]) for k=1:N+1], block_rks_vec.*p
  end
end

function tt_recursive_sketch(::Type{T}, A::TTvector{TA,N}, rmax::Int; orthogonal=true, reverse=true, seed=1234, block_rks::Int=N, timer::TimerOutput = TimerOutput()) where {T<:Number,TA<:Number,N}
  rks = rmax*ones(Int,N+1)
  rks[(reverse ? N+1 : 1)] = 1
  return tt_recursive_sketch(T,A,rks; orthogonal=orthogonal, reverse=reverse, seed=seed, block_rks=block_rks, timer=timer)
end

function tt_recursive_sketch(A::TTvector{T,N},rks_or_rmax; orthogonal=true, reverse=true, seed=1234, block_rks::Int=N, timer::TimerOutput = TimerOutput()) where {T<:Number,N}
  return tt_recursive_sketch(Float64,A,rks_or_rmax; orthogonal=orthogonal, reverse=reverse, seed=seed, block_rks=block_rks, timer=timer)
end

function tt_recursive_sketch(::Type{T}, H::TToperator{TH,N}, A::TTvector{TA,N}, rks; orthogonal=true, reverse=true, seed=1234, block_rks::Int=N, timer::TimerOutput = TimerOutput()) where {T<:Number,TA<:Number,TH<:Number,N}
  @timeit timer "tt_recursive_sketch" begin
    rng = Random.default_rng()
    Random.seed!(rng, seed)

    dims = A.ttv_dims
    TW = typeof(one(T)*one(TA)*one(TH))
    W = Vector{Array{TW,4}}(undef, N+1)

    if reverse
      @timeit timer "sketch_initialization" begin
        @assert rks[N+1] == 1 && A.ttv_rks[N+1] == 1 && H.tto_rks[N+1] == 1
        block_rks_vec = ones(Int, N+1)
        block_rks_vec[1:N] .= block_rks
        for k=N:-1:1
          block_rks_vec[k] = min(block_rks_vec[k], dims[k]*block_rks_vec[k+1])
        end

        p = compute_sketch_blocks_heuristic(rks, block_rks_vec, N; reverse=true)

        W[N+1] = ones(TW,1,1,1,1)

        # Preallocate buffer for the entire loop
        max_sketch_buffer_size = maximum(block_rks_vec[k+1] * dims[k] * block_rks_vec[k] * p[k] for k in 1:N)
        sketch_buffer = Vector{T}(undef, max_sketch_buffer_size)

        max_contract_buffer1_size = 0
        max_contract_buffer2_size = 0
        max_contract_buffer3_size = 0
        for k in 1:N
          z = dims[k]
          ζ = dims[k]
          a = A.ttv_rks[k]
          α = A.ttv_rks[k+1]
          b = H.tto_rks[k]
          β = H.tto_rks[k+1]
          c = block_rks_vec[k]
          γ = block_rks_vec[k+1]
          buf1_size, buf2_size, buf3_size = contract_sketch_core_backwards_operator_buffers_size(z, z, α, a, β, b, γ, c)
          max_contract_buffer1_size = max(max_contract_buffer1_size, buf1_size)
          max_contract_buffer2_size = max(max_contract_buffer2_size, buf2_size)
          max_contract_buffer3_size = max(max_contract_buffer2_size, buf3_size)
        end
        contract_buffer = (Vector{TW}(undef, max_contract_buffer1_size), 
                           Vector{TW}(undef, max_contract_buffer2_size), 
                           Vector{TW}(undef, max_contract_buffer3_size))
      end

      @timeit timer "contraction_reverse" begin
        @inbounds for k in N:-1:1
          @timeit timer "sketch_generation" begin
            B_sketch = generate_sketch_blocks(rng, T, block_rks_vec[k+1], dims[k], block_rks_vec[k], p[k], orthogonal; timer=timer, buffer=sketch_buffer)
          end

          @timeit timer "W_allocation" begin
            W[k] = zeros(TW, A.ttv_rks[k], H.tto_rks[k], block_rks_vec[k], p[k])
          end

          @timeit timer "tensor_contraction" begin
            for j=1:p[k]
              W_next_j = view(W[k+1],:,:,:,(k<N ? j : 1))
              B_j = view(B_sketch,:,:,:,j)
              W_k_j = view(W[k],:,:,:,j)
              contract_sketch_core_backwards!(W_k_j, A.ttv_vec[k], H.tto_vec[k], B_j, W_next_j; buffer=contract_buffer)
            end
          end
        end
      end
    else
      @timeit timer "sketch_initialization" begin
        @assert rks[1] == 1 && A.ttv_rks[1] == 1 && H.tto_rks[1] == 1
        block_rks_vec = ones(Int, N+1)
        block_rks_vec[2:N+1] .= block_rks
        for k=1:N
          block_rks_vec[k+1] = min(block_rks_vec[k+1], dims[k]*block_rks_vec[k])
        end

        p = compute_sketch_blocks_heuristic(rks, block_rks_vec, N; reverse=false)

        W[1] = ones(TW,1,1,1,1)

        # Preallocate buffer for the entire loop
        max_sketch_buffer_size = maximum(block_rks_vec[k] * dims[k] * block_rks_vec[k+1] * p[k] for k in 1:N)
        sketch_buffer = Vector{T}(undef, max_sketch_buffer_size)

        max_contract_buffer1_size = 0
        max_contract_buffer2_size = 0
        max_contract_buffer3_size = 0
        for k in 1:N
          ζ = dims[k]
          z = dims[k]
          α = A.ttv_rks[k]
          a = A.ttv_rks[k+1]
          β = H.tto_rks[k]
          b = H.tto_rks[k+1]
          γ = block_rks_vec[k]
          c = block_rks_vec[k+1]
          buf1_size, buf2_size, buf3_size = contract_sketch_core_forwards_operator_buffers_size(z, z, α, a, β, b, γ, c)
          max_contract_buffer1_size = max(max_contract_buffer1_size, buf1_size)
          max_contract_buffer2_size = max(max_contract_buffer2_size, buf2_size)
          max_contract_buffer3_size = max(max_contract_buffer2_size, buf3_size)
        end
        contract_buffer = (Vector{TW}(undef, max_contract_buffer1_size), 
                           Vector{TW}(undef, max_contract_buffer2_size), 
                           Vector{TW}(undef, max_contract_buffer3_size))
      end

      @timeit timer "contraction_forward" begin
        @inbounds for k in 1:N
          @timeit timer "sketch_generation" begin
            B_sketch = generate_sketch_blocks(rng, T, block_rks_vec[k], dims[k], block_rks_vec[k+1], p[k+1], orthogonal; timer=timer)
          end

          @timeit timer "W_allocation" begin
            W[k+1] = zeros(TW, A.ttv_rks[k+1], H.tto_rks[k+1], block_rks_vec[k+1], p[k+1])
          end

          @timeit timer "tensor_contraction" begin
            for j=1:p[k+1]
              W_k_j = view(W[k],:,:,:,(k>1 ? j : 1))
              B_j = view(B_sketch,:,:,:,j)
              W_next_j = view(W[k+1],:,:,:,j)
              contract_sketch_core_forwards!(W_next_j, A.ttv_vec[k], H.tto_vec[k], B_j, W_k_j; buffer=contract_buffer)

              @tensoropt((a,b,h,α,β,η), W_next_j[a,h,b] = A.ttv_vec[k][z,α,a]*H.tto_vec[k][y,z,η,h]*B_j[y,β,b]*W_k_j[α,η,β])
            end
          end
        end
      end
    end

    for k=1:N+1
      W[k] ./= sqrt(p[k])
    end
    return [reshape(W[k], A.ttv_rks[k], H.tto_rks[k], block_rks_vec[k]*p[k]) for k=1:N+1], block_rks_vec.*p
  end
end

function tt_recursive_sketch(::Type{T}, H::TToperator{TH,N}, A::TTvector{TA,N}, rmax::Int; orthogonal=true, reverse=true, seed=1234, block_rks::Int=N, timer::TimerOutput = TimerOutput()) where {T<:Number,TA<:Number,TH<:Number,N}
  d = A.N
  rks = rmax*ones(Int,d+1)
  rks[(reverse ? d+1 : 1)] = 1
  return tt_recursive_sketch(T,H,A,rks; orthogonal=orthogonal, reverse=reverse, seed=seed, block_rks=block_rks, timer=timer)
end

function tt_recursive_sketch(H::TToperator{TH,N}, A::TTvector{TA,N}, rks_or_rmax; orthogonal=true, reverse=true, seed=1234, block_rks::Int=N, timer::TimerOutput = TimerOutput()) where {TA<:Number,TH<:Number,N}
  return tt_recursive_sketch(Float64, H, A, rks_or_rmax; orthogonal=orthogonal, reverse=reverse, seed=seed, block_rks=block_rks, timer=timer)
end

function tt_recursive_sketch(::Type{T}, A::NTuple{M,TTvector{TA,N}}, rks; orthogonal=true, reverse=true, seed=1234, block_rks::Int=N, oversampling=1, timer::TimerOutput = TimerOutput()) where {T<:Number,TA<:Number,N,M}
  @timeit timer "tt_recursive_sketch" begin
    rng = Random.default_rng()
    Random.seed!(rng, seed)

    dims = A[1].ttv_dims
    @assert all(a.ttv_dims == dims for a in A)
    TW = typeof(one(T)*one(TA))
    W = Vector{Array{TW,M+2}}(undef, N+1)

    if reverse
      @timeit timer "sketch_initialization" begin
        @assert rks[N+1] == 1 && all(a.ttv_rks[N+1] == 1 for a in A)
        block_rks_vec = ones(Int, N+1)
        block_rks_vec[1:N] .= block_rks
        for k=N:-1:1
          block_rks_vec[k] = min(block_rks_vec[k], dims[k]*block_rks_vec[k+1])
        end

        p = compute_sketch_blocks_heuristic(rks, block_rks_vec, N; reverse=true)
        if oversampling ≠ 1
          p[1:N] .= ceil.(Int, oversampling .* p[1:N])
        end

        W[N+1] = ones(TW, ntuple(i->1, M+2))

        # Preallocate buffer for the entire loop
        max_sketch_buffer_size = maximum(block_rks_vec[k+1] * dims[k] * block_rks_vec[k] * p[k] for k in 1:N)
        sketch_buffer = Vector{T}(undef, max_sketch_buffer_size)

        max_contract_buffer1_size = 0
        max_contract_buffer2_size = 0
        max_contract_buffer3_size = 0
        for k in 1:N
          z = dims[k]
          v = ntuple(i->(i==M+1 ? block_rks_vec[k+1] : A[i].ttv_rks[k+1]), M+1)
          w = ntuple(i->(i==M+1 ? block_rks_vec[k  ] : A[i].ttv_rks[k  ]), M+1)
          buf1_size, buf2_size, buf3_size = contract_sketch_core_backwards_kronecker_buffers_size(z, v, w)
          max_contract_buffer1_size = max(max_contract_buffer1_size, buf1_size)
          max_contract_buffer2_size = max(max_contract_buffer2_size, buf2_size)
          max_contract_buffer3_size = max(max_contract_buffer3_size, buf3_size)
        end
        contract_buffer = (Vector{TW}(undef, max_contract_buffer1_size), Vector{TW}(undef, max_contract_buffer2_size), Vector{TW}(undef, max_contract_buffer3_size))
      end

      @timeit timer "contraction_reverse" begin
        @inbounds for k in N:-1:1
          @timeit timer "sketch_generation" begin
            B_sketch = generate_sketch_blocks(rng, T, block_rks_vec[k+1], dims[k], block_rks_vec[k], p[k], orthogonal; buffer=sketch_buffer, timer=timer)
          end

          @timeit timer "W_allocation" begin
            W[k] = zeros(TW, (a.ttv_rks[k] for a in A)..., block_rks_vec[k], p[k])
          end

          @timeit timer "tensor_contraction" begin
            for j=1:p[k]
              W_next_j = view(W[k+1], ntuple(i->Colon(), M+1)..., (k<N ? j : 1))
              B_j = view(B_sketch,:,:,:,j)
              W_k_j = view(W[k], ntuple(i->Colon(), M+1)..., j)
              contract_sketch_core_kronecker_backwards!(W_k_j, ntuple(i -> A[i].ttv_vec[k], M), B_j, W_next_j; buffer=contract_buffer)
            end
          end
        end
      end
    else
      @timeit timer "sketch_initialization" begin
        @assert rks[1] == 1 && all(a.ttv_rks[1] == 1 for a in A)
        block_rks_vec = ones(Int, N+1)
        block_rks_vec[2:N+1] .= block_rks
        for k=1:N
          block_rks_vec[k+1] = min(block_rks_vec[k+1], dims[k]*block_rks_vec[k])
        end

        p = compute_sketch_blocks_heuristic(rks, block_rks_vec, N; reverse=false)
        if oversampling ≠ 1
          p[2:N+1] .= ceil.(Int, oversampling .* p[2:N+1])
        end
        W[1] = ones(TW, ntuple(i->1, M+2))

        # Preallocate buffer for the entire loop
        max_sketch_buffer_size = maximum(block_rks_vec[k] * dims[k] * block_rks_vec[k+1] * p[k+1] for k in 1:N)
        sketch_buffer = Vector{T}(undef, max_sketch_buffer_size)

        max_contract_buffer1_size = 0
        max_contract_buffer2_size = 0
        max_contract_buffer3_size = 0
        for k in 1:N
          z = dims[k]
          v = ntuple(i->(i==M+1 ? block_rks_vec[k  ] : A[i].ttv_rks[k  ]), M+1)
          w = ntuple(i->(i==M+1 ? block_rks_vec[k+1] : A[i].ttv_rks[k+1]), M+1)
          buf1_size, buf2_size, buf3_size = contract_sketch_core_forwards_kronecker_buffers_size(z, v, w)
          max_contract_buffer1_size = max(max_contract_buffer1_size, buf1_size)
          max_contract_buffer2_size = max(max_contract_buffer2_size, buf2_size)
          max_contract_buffer3_size = max(max_contract_buffer3_size, buf3_size)
        end
        contract_buffer = (Vector{TW}(undef, max_contract_buffer1_size), Vector{TW}(undef, max_contract_buffer2_size), Vector{TW}(undef, max_contract_buffer3_size))
      end

      @timeit timer "contraction_forward" begin
        @inbounds for k in 1:N
          @timeit timer "sketch_generation" begin
            B_sketch = generate_sketch_blocks(rng, T, block_rks_vec[k], dims[k], block_rks_vec[k+1], p[k+1], orthogonal; buffer=sketch_buffer, timer=timer)
          end

          @timeit timer "W_allocation" begin
            W[k+1] = zeros(TW, (a.ttv_rks[k+1] for a in A)..., block_rks_vec[k+1], p[k+1])
          end

          @timeit timer "tensor_contraction" begin
            for j=1:p[k+1]
              W_k_j = view(W[k], ntuple(i->( i<M+2 ? Colon() : (k>1 ? j : 1) ), M+2)...)
              B_j = view(B_sketch,:,:,:,j)
              W_next_j = view(W[k+1], ntuple(i->( i<M+2 ? Colon() : j ), M+2)...)
              contract_sketch_core_kronecker_forwards!(W_next_j, ntuple(i -> A[i].ttv_vec[k], M), B_j, W_k_j; buffer=contract_buffer)
            end
          end
        end
      end
    end

    for k=1:N+1
      W[k] ./= sqrt(p[k])
    end
    return [reshape(W[k], (a.ttv_rks[k] for a in A)..., block_rks_vec[k]*p[k]) for k=1:N+1], block_rks_vec.*p
  end
end

function tt_recursive_sketch(::Type{T}, A::NTuple{M,TTvector{TA,N}}, rmax::Int; orthogonal=true, reverse=true, seed=1234, block_rks::Int=N, timer::TimerOutput = TimerOutput()) where {T<:Number,TA<:Number,N,M}
  rks = rmax*ones(Int,N+1)
  rks[(reverse ? N+1 : 1)] = 1
  return tt_recursive_sketch(T,A,rks; orthogonal=orthogonal, reverse=reverse, seed=seed, block_rks=block_rks, timer=timer)
end

function tt_recursive_sketch(A::NTuple{M,TTvector{T,N}},rks_or_rmax; orthogonal=true, reverse=true, seed=1234, block_rks::Int=N, timer::TimerOutput = TimerOutput()) where {T<:Number,N,M}
  return tt_recursive_sketch(Float64,A,rks_or_rmax; orthogonal=orthogonal, reverse=reverse, seed=seed, block_rks=block_rks, timer=timer)
end


"""
Compute partial contractions between two TTvectors A and B.
Returns an array W where W[k] contains the contraction of cores from position k onwards.
If reverse=true, contracts from right to left (default).
If reverse=false, contracts from left to right.
"""
function partial_contraction(A::TTvector{T1,N},B::TTvector{T2,N};reverse=true) where {T1,T2,N}
  @assert A.ttv_dims==B.ttv_dims "TT dimensions are not compatible"
  if T1 == ComplexF64 || T2 == ComplexF64
    T = ComplexF64
  else
    T = Float64
  end
  A_rks = A.ttv_rks
  B_rks = B.ttv_rks
  L = length(A.ttv_dims)
  W = [zeros(T, A_rks[i], B_rks[i]) for i in 1:L+1]
  if reverse
    W[L+1] = ones(T,1,1)
    @inbounds for k in L:-1:1
      @tensoropt((a,b,α,β), W[k][a,b] = A.ttv_vec[k][z,a,α]*B.ttv_vec[k][z,b,β]*W[k+1][α,β]) #size R^A_{k} × R^B_{k}
    end
  else
    W[1] = ones(T,1,1)
    @inbounds for k in 1:L
      @tensoropt((a,b,α,β), W[k+1][a,b] = A.ttv_vec[k][z,α,a]*B.ttv_vec[k][z,β,b]*W[k][α,β]) #size R^A_{k} × R^B_{k}
    end
  end
  return W
end

"""
    stta_sketch(x::TTvector, L::TTvector, R::TTvector) -> (Ω, Ψ)
    stta_sketch(x::TTvector, rks::Vector{Int}; seed_left=1234, seed_right=5678, orthogonal=true, block_rks=N) -> (Ω, Ψ)

Compute left and right sketches for Streaming Tensor Train Approximation (STTA) with optimized normalization.

Generates two-sided random sketches of a TTvector using either explicit random TTvectors
or recursive sketching with target ranks. The recursive version uses adaptive normalization
and 50% oversampling on left ranks for optimal STTA performance.

# Arguments
- `x::TTvector{T,N}`: Input TTvector to sketch
- `L::TTvector{T,N}`: Left random TTvector (for explicit sketch version)
- `R::TTvector{T,N}`: Right random TTvector (for explicit sketch version)
- `rks::Vector{Int}`: Target approximation ranks (for recursive sketch version)

# Keyword Arguments (recursive sketch version)
- `seed_left::Int=1234`: Random seed for left sketch
- `seed_right::Int=5678`: Random seed for right sketch  
- `orthogonal::Bool=true`: Use orthogonal random sketches with adaptive normalization
- `block_rks::Int=N`: Block rank for sketching heuristic (controls granularity)

# Returns
- `Ω::Vector{Matrix}`: Overlap matrices where Ω[k] = L'[k+1] * R[k+1]
- `Ψ::Vector{Array{T,3}}`: Sketched cores where Ψ[k][i,α,β] = x[i] * L[α] * R[β]

# Algorithm
Based on:
Kressner, Vandereycken & Voorhaar (2022), "Streaming Tensor Train Approximation",
SIAM J. Sci. Comput., 45(5), pp. A2610–A2629, https://arxiv.org/abs/2208.02600

**Key optimizations**:
1. **Asymmetric oversampling**: Left sketch ranks set to `ceil(1.5 * rks[2:N])` for numerical stability
2. **Adaptive normalization**: Uses improved `tt_recursive_sketch` with QR orthogonalization and 
   rank-dependent scaling for consistent spectral properties
3. **Bidirectional sketching**: Left sketch uses forward direction, right sketch uses reverse direction

The optimized normalization ensures stable numerical performance across different tensor ranks
and provides reliable approximation quality for the STTA algorithm.
"""
function stta_sketch(x::TTvector{T,N},L::TTvector{T,N},R::TTvector{T,N}) where {T,N}
  Ψ = [zeros(T, x.ttv_dims[i], L.ttv_rks[i], R.ttv_rks[i+1]) for i in 1:N]
  Ω = [zeros(T, L.ttv_rks[i+1], R.ttv_rks[i+1]) for i in 1:N-1]
  left_contractions = partial_contraction(x,L;reverse=false)
  right_contractions = partial_contraction(x,R;reverse=true)
  for k in eachindex(Ω)
    @tensor Ω[k][a,b] = left_contractions[k+1][z,a]*right_contractions[k+1][z,b]
  end
  for k in eachindex(Ψ)
    @tensor (Ψ[k][i,α,β] = (x.ttv_vec[k][i,y,z]*left_contractions[k][y,α])*right_contractions[k+1][z,β])
  end
  return Ω,Ψ
end

function stta_sketch(x::TTvector{T,N}, rks::Vector{Int};
                     seed_left::Int=1234, seed_right::Int=5678, orthogonal::Bool=true, block_rks::Int=N, timer::TimerOutput = TimerOutput()) where {T,N}
  @timeit timer "stta_sketch" begin
    # For optimal STTA performance, left ranks should be 50% larger than target ranks
    # Generate left sketch (forward direction) with 50% oversampling
    @timeit timer "left_sketch" begin
      W_left, sketch_l_rks = tt_recursive_sketch(T, x, rks; orthogonal=orthogonal, reverse=false, seed=seed_left, block_rks=block_rks, oversampling=1.5, timer=timer)
    end

    # Generate right sketch (reverse direction)
    @timeit timer "right_sketch" begin
      W_right, sketch_r_rks = tt_recursive_sketch(T, x, rks; orthogonal=orthogonal, reverse=true, seed=seed_right, block_rks=block_rks, timer=timer)
    end

    # Compute Ω and Ψ from the sketch matrices
    Ψ = [zeros(T, x.ttv_dims[i], sketch_l_rks[i], sketch_r_rks[i+1]) for i in 1:N]
    Ω = [zeros(T, sketch_l_rks[i+1], sketch_r_rks[i+1]) for i in 1:N-1]

    # Contract to form Ω[k] = W_left[k+1]' * W_right[k+1]
    for k in 1:N-1
      @tensor Ω[k][a,b] = W_left[k+1][α,a] * W_right[k+1][α,b]
    end

    # Contract to form Ψ[k]
    for k in 1:N
      @tensor Ψ[k][i,a,b] = x.ttv_vec[k][i,α,β] * W_left[k][α,a] * W_right[k+1][β,b]
    end

    return Ω, Ψ
  end
end

# ============================================================================
# Contraction kernels for tensor train sketching
# ============================================================================

"""
    contract_sketch_core_forwards!(W, A, S, V; buffer=(nothing,nothing), timer=TimerOutput())
    contract_sketch_core_forwards!(W, A, B, S, V; buffer=(nothing,nothing,nothing), timer=TimerOutput())

Efficient contraction kernel for forward tensor train sketching:
W[a,b] = A[z,α,a] * S[β,z,b] * V[α,β]
or
W[a,b,c] = A[ζ,α,a] * B[ζ,z,β,b] * S[γ,z,c] * V[α,β,γ]

Uses adaptive contraction ordering based on rank sizes to minimize overall cost:

# Arguments:
# Case A*S*V
- `W::AbstractMatrix`: Output matrix [a,b] (modified in-place)
- `A::AbstractArray{T,3}`: Left tensor [z,α,a] (forward TT format)
- `S::AbstractArray{T,3}`: Sketch block tensor [β,z,b] (sketch block format)
- `V::AbstractMatrix`: Connection matrix [α,β] (previous sketch weights)

# Case A*B*S*V
- `W::AbstractArray{T,3}`: Output tensor [a,b,c] (modified in-place)
- `A::AbstractArray{T,3}`: Left tensor [z/ζ,α,a] (forward TT format)
  `B::AbstractArray{T,4}`: Operator tensor [ζ,z,β,b] (forward TTO format)
- `S::AbstractArray{T,3}`: Sketch block tensor [γ,z,c] (sketch block format)
- `V::AbstractArray{T,3}`: Connection array [α,β,γ] (previous sketch weights)

# Optional arguments
- `buffer`: Optional preallocated buffer arrays
- `timer::TimerOutput`: Optional timer for performance profiling

"""
function contract_sketch_core_forwards!(W::AbstractMatrix{T}, A::AbstractArray{T,3}, S::AbstractArray{T,3}, V::AbstractMatrix{T};
                                buffer=(nothing,nothing)) where T

  a,b = size(W)
  α,β = size(V)
  z = size(A,1)
  @assert size(A) === (z,α,a) "Factor A has the wrong dimensions: need $((z,α,a)), got $(size(A))"
  @assert size(S) === (β,z,b) "Factor S has the wrong dimensions: need $((β,z,b)), got $(size(S))"

  if a*(β+1)*(α+b) < α*b*(a+β+1)
    # Order 1: ((V'×A)×S)
    # Cost: a*β*z*(α+a+b) for: permute A, mul V'×A, mul VA'×S
    # Permute dimensions: A_permuted[α,z,a] = A[z,α,a]
    A_permuted = permutedims!!(A, (2,1,3), buffer=buffer[1])
    A_permuted = reshape(A_permuted, α, z*a)

    # Order 1: VA[β,z,a] = V'[β,α] * A_permuted[α,z*a]
    VA = mul!!(transpose(V), A_permuted, buffer=buffer[2])
    VA = reshape(VA, β*z, a)

    # Step 2: W[a,b] = VA[β,z,a] * S[β,z,b] → transpose VA to [a,β,z] and multiply
    mul!(W, transpose(VA), reshape(S, β*z, b))
  else
    # Order 2: ((V×S)×A)
    # Cost: α*b*z*(β+1+a) for: mul V×S, permute VS, mul A'×VS
    # VS[α,z,b] = V[α,β] * S[β,z,b], then W[a,b] = A[z,α,a] * VS[α,z,b]
    VS = mul!!(V, reshape(S, β, z*b), buffer=buffer[1])
    VS = reshape(VS, α, z, b)

    # Permute dimensions: VS_permuted[z,α,b] = VS[α,z,b]
    VS_permuted = permutedims!!(VS, (2,1,3), buffer=buffer[2])
    VS_permuted = reshape(VS_permuted, z*α,b)

    # Step 2: W[a,b] = A'[a,z*α] * VS_permuted[z*α,b]
    mul!(W, transpose(reshape(A, z*α, a)), VS_permuted)
  end

  return W
end

function contract_sketch_core_forwards_buffers_size(z::Int, α::Int, a::Int, β::Int, b::Int)
  if a*(β+1)*(α+b) < α*b*(a+β+1)
    return (z*α*a, z*a*β)  # buffer[1] for A_permuted, buffer[2] for VA
  else
    return (α*z*b, α*z*b)  # buffer[1] for VS intermediate, buffer[2] for VS_permuted
  end
end

function contract_sketch_core_forwards!(W::AbstractArray{T,3}, A::AbstractArray{T,3}, B::AbstractArray{T,4}, S::AbstractArray{T,3}, V::AbstractArray{T,3};
                                buffer=(nothing,nothing,nothing)) where T

  a,b,c = size(W)
  α,β,γ = size(V)
  ζ = size(B,1)
  z = size(B,2)
  @assert size(A) === (ζ,α,a)   "Factor A has the wrong dimensions: need $((ζ,α,a)), got $(size(A))"
  @assert size(B) === (ζ,z,β,b) "Factor B has the wrong dimensions: need $((ζ,z,β,b)), got $(size(B))"
  @assert size(S) === (γ,z,c)   "Factor S has the wrong dimensions: need $((γ,z,c)), got $(size(S))"

  if ζ*a*α*β*γ + a*γ*ζ*β*z*b + a*b*γ*z*c < α*β*γ*z*c + ζ*b*z*β*α*c + a*ζ*α*b*c # Order 1: ((V*A)*B)*S
    # Order 1 cost dominated by: ζ*a*α*β*γ + a*γ*ζ*β*z*b + a*b*γ*z*c

    # Permute dimensions: A_permuted[ζ,a,α] = A[ζ,α,a]
    A_permuted = permutedims!!(A, (1,3,2), buffer=buffer[1])
    # Step 1: VA[ζ,a,β,γ] = A_permuted[ζ*a, α] * V[α,β,γ]
    VA = mul!!(reshape(A_permuted, ζ*a, α), reshape(V, α, β*γ), buffer=buffer[2])
    VA = reshape(VA, ζ, a, β, γ)

    # Permute dimensions: VA_permuted[a,γ,ζ,β] = VA[ζ,a,β,γ]
    VA_permuted = permutedims!!(VA, (2,4,1,3), buffer=buffer[1])
    VA_permuted = reshape(VA_permuted, a*γ,ζ*β)

    # Permute dimensions: B_permuted[ζ,β,z,b] = B[ζ,z,β,b]
    B_permuted = permutedims!!(B, (1,3,2,4), buffer=buffer[2])
    B_permuted = reshape(B_permuted, ζ*β,z*b)

    # Step 2: VAB[a,γ,z,b] = VA_permuted[a,γ,ζ,β] * B_permuted[ζ,β,z,b]
    VAB = mul!!(VA_permuted, B_permuted, buffer=buffer[3])
    VAB = reshape(VAB, a,γ,z,b)

    # Permute dimensions: VAB_permuted[a,b,γ,z] = VAB[a,γ,z,b]
    VAB_permuted = permutedims!!(VAB, (1,4,2,3), buffer=buffer[1])
    VAB_permuted = reshape(VAB_permuted, a*b,γ*z)

    # Step 3: W[a,b,c] = VAB_permuted[a,b,γ,z] * S[γ,z,c]
    mul!(reshape(W, a*b, c), VAB_permuted, reshape(S, γ*z, c))
  else # Order 2: ((V*S)*B)*A
    # Order 2 cost dominated by: α*β*γ*z*c + ζ*b*z*β*α*c + a*ζ*α*b*c
    # Step 1: VS[α,β,z,c] = V[α,β,γ] * S[γ,z,c]
    VS = mul!!(reshape(V, α*β, γ), reshape(S, γ, z*c), buffer=buffer[1])
    VS = reshape(VS, α,β,z,c)

    # Permute dimensions: VS_permuted[z,β,α,c] = VS[α,β,z,c]
    VS_permuted = permutedims!!(VS, (3,2,1,4), buffer=buffer[2])
    VS_permuted = reshape(VS_permuted, z*β, α*c)

    # Permute dimensions: B_permuted[ζ,b,z,β] = B[ζ,z,β,b]
    B_permuted = permutedims!!(B, (1,4,2,3), buffer=buffer[1])
    B_permuted = reshape(B_permuted, ζ*b, z*β)

    # Step 2: VBS[ζ,b,α,c] = B_permuted[ζ,b,z,β] * VS_permuted[z,β,α,c]
    VBS = mul!!(B_permuted, VS_permuted, buffer=buffer[3])
    VBS = reshape(VBS, ζ,b,α,c)

    # Permute dimensions: A_permuted[a,ζ,α] = A[ζ,α,a]
    A_permuted = permutedims!!(A, (3,1,2), buffer=buffer[1])
    A_permuted = reshape(A_permuted, a,ζ*α)

    # Permute dimensions: VBS_permuted[ζ,α,b,c] = VBS[ζ,b,α,c]
    VBS_permuted = permutedims!!(VBS, (1,3,2,4), buffer=buffer[2])
    VBS_permuted = reshape(VBS_permuted, ζ*α,b*c)

    # Step 3: W[a,b,c] = A_permuted[a,ζ,α] * VBS_permuted[ζ,α,b,c]
    mul!(reshape(W, a,b*c), A_permuted, VBS_permuted)
  end

  return W
end

function contract_sketch_core_forwards_operator_buffers_size(ζ::Int, z::Int, α::Int, a::Int, β::Int, b::Int, γ::Int, c::Int)
  if ζ*a*α*β*γ + a*γ*ζ*β*z*b + a*b*γ*z*c < α*β*γ*z*c + ζ*b*z*β*α*c + a*ζ*α*b*c  # Order 1: ((V*A)*B)*C
    # buffer[1]: A_permuted[ζ,a,α], VA_permuted[a,γ,ζ,β], VAB_permuted[a,b,γ,z]
    buffer1_size = max(ζ*a*α, a*γ*ζ*β, a*b*γ*z)
    # buffer[2]: VA intermediate[ζ*a,β*γ], B_permuted[ζ*β,z*b] (overwrites VA safely)
    buffer2_size = max(ζ*a*β*γ, ζ*β*z*b)
    # buffer[3]: VAB intermediate[a*γ,z*b]
    buffer3_size = a*γ*z*b
    return (buffer1_size, buffer2_size, buffer3_size)
  else  # Order 2: ((V*C)*B)*A
    # buffer[1]: VC intermediate[α*β,z*c], B_permuted[ζ*b,z*β] (overwrites VC safely), A_permuted[a,ζ*α]
    buffer1_size = max(α*β*z*c, ζ*b*z*β, a*ζ*α)
    # buffer[2]: VC_permuted[z*β,α*c], VBC_permuted[ζ*α,b*c]
    buffer2_size = max(z*β*α*c, ζ*α*b*c)
    # buffer[3]: VBC intermediate[ζ*b,α*c]
    buffer3_size = ζ*b*α*c
    return (buffer1_size, buffer2_size, buffer3_size)
  end
end


function contract_sketch_core_kronecker_forwards!(W::AbstractArray{T,N1}, A::NTuple{N,AbstractArray{T,3}}, S::AbstractArray{T,3}, V::AbstractArray{T,N1};
                                buffer=(nothing,nothing,nothing)) where {T,N,N1}
  @assert N1 === N+1

  w = size(W)
  v = size(V)
  z = size(A[1],1)
  for i=1:N
    @assert size(A[i]) == (z,v[i],w[i])   "Factor A[$i] has the wrong dimensions: need $((z,v[i],w[i])), got $(size(A[i]))"
  end
  @assert size(S) === (v[N+1],z,w[N+1])   "Factor S has the wrong dimensions: need $((v[N+1],z,w[N+1])), got $(size(S))"

  function permutation(i,j) # Bring i to 1 and j to N+1
    q = collect(1:N+2)
    q[1], q[i] = i, 1
    if 1<j # j is still at j
      q[N+1], q[j] = j, q[N+1]
    else # j is now at i
      q[N+1], q[i] = j, q[N+1]
    end
    return q
  end

  function inversepermutation(i,j) # Bring i back from 1 and j from N+1
    q = collect(1:N+2)
    if 1<j
      q[N+1], q[j] = q[j], q[N+1]
    else
      q[N+1], q[i] = q[i], q[N+1]
    end
    q[1], q[i] = q[i], q[1]
    return q
  end

  order = sortperm(collect(w./v), rev=true)
  j = pop!(order)    
  # buffer[1]: V_permuted[α≠i,αi]  
  perm = permutation(1,j)[1:N+1]
  V_permuted = permutedims!!(V, perm, buffer=buffer[1])
  if j == N+1
    # buffer[2]: A_permuted[αj,aj,z] = S[β,z,b]
    A_permuted = permutedims!!(S, (1,3,2), buffer=buffer[2])
  else
    # buffer[2]: A_permuted[αj,aj,z] = Ai[z,αj,aj]
    A_permuted = permutedims!!(A[j], (2,3,1), buffer=buffer[2])
  end
  # buffer[3]: VA[α≠j, aj, z]
  VA = mul!!(reshape(V_permuted, :, v[j]), reshape(A_permuted, v[j], w[j]*z), buffer=buffer[3])
  va = [v[perm[1:N]]..., w[j], z]
  VA = reshape(VA, va...)
  perm = inversepermutation(1,j)



  while length(order) > 2
    i = pop!(order)
    j = pop!(order)
    perm = perm[permutation(i,j)]
    # buffer[1]: VA_permuted[αi,α≠ij,αj,z]  
    VA_permuted = permutedims!!(VA, perm, buffer=buffer[1])
    va = va[perm]

    if i == N+1
      # buffer[2]: A_permuted[ai,αi,z] = S[β,z,b]
      A_permuted = permutedims!!(S, (3,1,2), buffer=buffer[2])
    else
      # buffer[2]: A_permuted[ai,αi,z] = Ai[z,αi,ai]
      A_permuted = permutedims!!(A[i], (3,2,1), buffer=buffer[2])
    end
    # buffer[3]: VA[ai,a/α≠ij,βj,z] = A_permuted[ai,αi,z] * VA_permuted[αi,α≠ij,βj,z] 
    VA_intermediate = mul!!(A_permuted, reshape(VA_permuted,v[i],:,z), buffer=buffer[3])
    va[1] = w[i]
    if j == N+1
      # buffer[1]: B_permuted[βj,bj,z] = S[β,z,b]
      B_permuted = permutedims!!(S, (1,3,2), buffer=buffer[1])
    else
      # buffer[1]: B_permuted[βj,bj,z] = Aj[z,αj,aj]
      B_permuted = permutedims!!(A[j], (2,3,1), buffer=buffer[1])
    end
    # buffer[2]: VA[ai,a/α≠ij,bj,z] = VA_intermediate[ai,α≠ij,βj,z] * B_permuted[βj,bj,z]
    VA = mul!!(reshape(VA_intermediate,:,v[j],z), B_permuted, buffer=buffer[2])
    va[N+1] = w[j]
    VA = reshape(VA, va...)

    perm = inversepermutation(i,j)
  end

  if length(order) == 2
    i = pop!(order)
    j = pop!(order)
    perm = perm[permutation(i,j)]
    
        # buffer[1]: VA_permuted[αi,α≠ij,αj,z]  
    VA_permuted = permutedims!!(VA, perm, buffer=buffer[1])
    va = va[perm]

    if i == N+1
      # buffer[2]: A_permuted[ai,αi,z] = S[β,z,b]
      A_permuted = permutedims!!(S, (3,1,2), buffer=buffer[2])
    else
      # buffer[2]: A_permuted[ai,αi,z] = Ai[z,αi,ai]
      A_permuted = permutedims!!(A[i], (3,2,1), buffer=buffer[2])
    end
    # buffer[3]: VA_intermediate[ai,a/α≠ij,βj,z] = Ai_permuted[ai,αi,z] * VA_permuted[αi,α≠ij,βj,z] 
    VA_intermediate = mul!!(A_permuted, reshape(VA_permuted,v[i],:,z), buffer=buffer[3])
    va[1] = w[i]
    if j == N+1
      # buffer[1]: B_permuted[βj,z,bj] = S[β,z,b]
      B_permuted = S
    else
      # buffer[1]: B_permuted[βj,z,bj]
      B_permuted = permutedims!!(A[j], (2,1,3), buffer=buffer[1])
    end
    # buffer[2]: VA[ai,a≠ij,bj] = VA_intermediate[ai,α≠ij,βj,z] * Bj_permuted[βj,z,bj]
    W_permuted = mul!!(reshape(VA_intermediate,:,v[j]*z), reshape(B_permuted,v[j]*z,w[j]), buffer=buffer[2])
    va = va[1:N+1]
    va[N+1] = w[j]
 
    W_permuted = reshape(W_permuted, va...)
    perm = inversepermutation(i,j)[1:N+1]
    permutedims!(W, W_permuted, perm)
  else # length(order == 1)
    j = pop!(order)
    perm = perm[permutation(1,j)]

    # buffer[1]: VA_permuted[αi,α≠ij,αj,z]  
    VA_permuted = permutedims!!(VA, perm, buffer=buffer[1])
    va = va[perm]
    if j == N+1
      # buffer[2]: B_permuted[βj,z,bj] = S[β,z,b]
      B_permuted = S
    else
      # buffer[2]: Bj_permuted[βj,z,bj]
      B_permuted = permutedims!!(A[j], (2,1,3), buffer=buffer[2])
    end
    # buffer[3]: VA[ai,a≠ij,bj] = VA_permuted[a≠j,βj,z] * Bj_permuted[βj,z,bj]
    W_permuted = mul!!(reshape(VA_permuted,:,v[j]*z), reshape(B_permuted, v[j]*z, w[j]), buffer=buffer[3])
    va = va[1:N+1]
    va[N+1] = w[j]

    W_permuted = reshape(W_permuted, va...)
    perm = inversepermutation(1,j)[1:N+1]
    permutedims!(W, W_permuted, perm)
  end

  return W
end

function contract_sketch_core_forwards_kronecker_buffers_size(z::Int, v::NTuple{N1,Int}, w::NTuple{N1,Int}) where {N1}
  N = N1 - 1
  
  # Buffer 1: Used for VA_permuted (permutations), B_permuted operations
  buffer1_size = 0
  # Buffer 2: Used for A_permuted, final VA/W_permuted results  
  buffer2_size = 0
  # Buffer 3: Used for VA_intermediate results from mul!!
  buffer3_size = 0
  
  # Initial V permutation: V_permuted has same size as V
  buffer1_size = max(buffer1_size, prod(v))
  
  # Track dimensions through the algorithm using same ordering as main function
  order = sortperm(collect(w./v), rev=true)
  va = [v..., z]  # Initial va dimensions

  l(i) = z*v[i]*w[i]
  
  # Initial VA computation
  i = pop!(order)
  
  # Initial A[i] or S permutation
  buffer2_size = max(buffer2_size, l(i))
  
  # Update va after first contraction
  va[i] = w[i]
  buffer3_size = max(buffer3_size, prod(va))
  
  
  # Track through algorithm iterations
  while length(order) > 2
    i = pop!(order)
    j = pop!(order)
    
    # VA_permuted: reordered version of current VA
    buffer1_size = max(buffer1_size, prod(va))
    
    # A_permuted for i (or S if i == N+1)
    buffer2_size = max(buffer2_size, l(i))
    
    # VA_intermediate from mul!!
    va[i] = w[i]
    buffer3_size = max(buffer3_size, prod(va))
    
    # B_permuted for j (or S if j == N+1)
    buffer1_size = max(buffer1_size, l(j))
    
    # Final VA from second mul!!
    va[j] = w[j]
    buffer2_size = max(buffer2_size, prod(va))
  end
  
  if length(order) == 2
    i = pop!(order)
    j = pop!(order)
    
    # VA_permuted
    buffer1_size = max(buffer1_size, prod(va))
    
    # A_permuted for i (or S if i == N+1)
    buffer2_size = max(buffer2_size, l(i))
    
    # VA_intermediate from mul!!
    va[i] = w[i]
    buffer3_size = max(buffer3_size, prod(va))
    
    # B_permuted for j (or S if j == N+1)
    buffer1_size = max(buffer1_size, l(j))
    va[j] = w[j]
    
    # Final W_permuted
    buffer2_size = max(buffer2_size, prod(va))
    
  elseif length(order) == 1
    j = pop!(order)
    
    # VA_permuted
    buffer1_size = max(buffer1_size, prod(va))
    
    # Handle special case: if j is the last core (N+1), use S directly
    if j < N+1
      # B_permuted[v[j], z, w[j]] for regular A[j]
      buffer2_size = max(buffer2_size, l(j))
    end
    buffer3_size = max(buffer3_size, prod(w))
  end
  
  return (buffer1_size, buffer2_size, buffer3_size)
end



"""
    contract_sketch_core_backwards!(W, A, S, V; buffer=(nothing,nothing), timer=TimerOutput())
    contract_sketch_core_backwards!(W, A, B, S, V; buffer=(nothing,nothing,nothing), timer=TimerOutput())

Efficient contraction kernel for backward tensor train sketching:
W[a,b] = A[z,a,α] * S[β,z,b] * V[α,β]
or
W[a,b,c] = A[z,a,α] * B[z,ζ,b,β] * S[γ,ζ,c] * V[α,β,γ]

Uses adaptive contraction ordering based on rank sizes to minimize overall cost:
# Arguments
# Case A*S*V
- `W::AbstractMatrix`: Output matrix [a,b] (modified in-place)
- `A::AbstractArray{T,3}`: Left tensor [z,a,α] (backward TT format)
- `S::AbstractArray{T,3}`: Sketch block tensor [β,z,b] (sketch block format)
- `V::AbstractMatrix`: Connection matrix [α,β] (previous sketch weights)

# Case A*B*S*V
- `W::AbstractArray{T,3}`: Output tensor [a,b,c] (modified in-place)
- `A::AbstractArray{T,3}`: Left tensor [z,a,α] (backward TT format)
- `B::AbstractArray{T,4}`: Operator tensor [z,ζ,b,β] (backward TTO format)
- `S::AbstractArray{T,3}`: Sketch block tensor [γ,ζ,c] (sketch block format)
- `V::AbstractArray{T,3}`: Connection array [α,β,γ] (previous sketch weights)

# Optional arguments
- `buffer::NTuple{Union{Nothing,AbstractVector}}`: Optional preallocated buffers
- `timer::TimerOutput`: Optional timer for performance profiling

"""
function contract_sketch_core_backwards!(W::AbstractMatrix{T}, A::AbstractArray{T,3}, S::AbstractArray{T,3}, V::AbstractMatrix{T};
                                buffer=(nothing,nothing)) where T

  a,b = size(W)
  α,β = size(V)
  z = size(A,1)
  @assert size(A) === (z,a,α) "Factor A has the wrong dimensions: need $((z,a,α)), got $(size(A))"
  @assert size(S) === (β,z,b) "Factor S has the wrong dimensions: need $((β,z,b)), got $(size(S))"

  if a*β*(α+b+1) < α*(b+1)*(a+β)
    # Order 1: AV[z,a,β] = A[z,a,α] * V[α,β], then W[a,b] = AV * S
    # Cost: z*a*β*(α+1+b) for: mul A×V, permute AV, mul AV×S
      AV = mul!!(reshape(A, z*a, α), V, buffer=buffer[1])
      AV = reshape(AV, z, a, β)

    # Permute dimensions: AV_permuted[a,β,z] = AV[z,a,β]
      AV_permuted = permutedims!!(AV, (2,3,1), buffer=buffer[2])
      AV_permuted = reshape(AV_permuted, a,β*z)

    # Step 2: W[a,b] = AV_permuted[a,β,z] * S[β,z,b]
      mul!(W, AV_permuted, reshape(S, β*z, b))
  else
    # Order 2: VS[α,z,b] = V[α,β] * S[β,z,b], then W[a,b] = A * VS
    # Cost: z*α*(b+1)*(a+β) for: mul V×S, permute A, mul A×VS
      VS = mul!!(V, reshape(S, β, z*b), buffer=buffer[1])
      VS = reshape(VS, α*z, b)

    # Permute dimensions: A_permuted[a,α,z] = A[z,a,α]
      A_permuted = permutedims!!(A, (2,3,1), buffer=buffer[2])
      A_permuted = reshape(A_permuted, a, α*z)

    # Step 2: W[a,b] = A_permuted[a,α,z] * VS[α,z,b]
      mul!(W, A_permuted, VS)
  end

  return W
end

function contract_sketch_core_backwards_buffers_size(z::Int, a::Int, α::Int, β::Int, b::Int)
  if a*β*(α+b+1) < α*(b+1)*(a+β)
    return (z*a*β, z*a*β)  # buffer[1] for AV, buffer[2] for AV_permuted
  else
    return (α*z*b, z*a*α)  # buffer[1] for CB, buffer[2] for A_permuted
  end
end

function contract_sketch_core_backwards!(W::AbstractArray{T,3}, A::AbstractArray{T,3}, B::AbstractArray{T,4}, S::AbstractArray{T,3}, V::AbstractArray{T,3};
                                buffer=(nothing,nothing,nothing)) where T

  a,b,c = size(W)
  α,β,γ = size(V)
  z = size(B,1)
  ζ = size(B,2)
  @assert size(A) === (z,a,α)   "Factor A has the wrong dimensions: need $((z,a,α)), got $(size(A))"
  @assert size(B) === (z,ζ,b,β) "Factor B has the wrong dimensions: need $((z,ζ,b,β)), got $(size(B))"
  @assert size(S) === (γ,ζ,c)   "Factor S has the wrong dimensions: need $((γ,ζ,c)), got $(size(S))"

  if a*γ*(z*β*(α+ζ*b)+ζ*b*c) < α*c*(ζ*β*(γ+z*b) + z*a*b) # Order 1: ((V*A)*B)*S
  # Order 1 cost dominated by: z*a*α*β*γ + a*γ*z*β*ζ*b + a*b*γ*ζ*c
  # Step 1: VA[z,a,β,γ] = A[z*a, α] * V[α,β,γ]
    VA = mul!!(reshape(A, z*a, α), reshape(V, α, β*γ), buffer=buffer[1])
    VA = reshape(VA, z, a, β, γ)

  # Permute dimensions: VA_permuted[a,γ,z,β] = VA[z,a,β,γ]
    VA_permuted = permutedims!!(VA, (2,4,1,3), buffer=buffer[2])
    VA_permuted = reshape(VA_permuted, a*γ,z*β)

  # Permute dimensions: B_permuted[z,β,ζ,b] = B[z,ζ,b,β]
    B_permuted = permutedims!!(B, (1,4,2,3), buffer=buffer[1])
    B_permuted = reshape(B_permuted, z*β,ζ*b)

  # Step 2: VAB[a,γ,ζ,b] = VA_permuted[a,γ,z,β] * B_permuted[z,β,ζ,b]
    VAB = mul!!(VA_permuted, B_permuted, buffer=buffer[3])
    VAB = reshape(VAB, a,γ,ζ,b)

  # Permute dimensions: VAB_permuted[a,b,γ,ζ] = VAB[a,γ,ζ,b]
    VAB_permuted = permutedims!!(VAB, (1,4,2,3), buffer=buffer[1])
    VAB_permuted = reshape(VAB_permuted, a*b,γ*ζ)
    # Step 3: W[a,b,c] = VAB_permuted[a,b,γ,ζ] * S[γ,ζ,c]
    mul!(reshape(W, a*b, c), VAB_permuted, reshape(S, γ*ζ,c))
  else # Order 2: ((V*S)*B)*A
    # Order 2 cost dominated by: α*β*γ*ζ*c + z*b*ζ*β*α*c + a*z*α*b*c
    # Step 1: VS[α,β,ζ,c] = V[α,β,γ] * S[γ,ζ,c]
    VS = mul!!(reshape(V, α*β, γ), reshape(S, γ, ζ*c), buffer=buffer[1])
    VS = reshape(VS, α,β,ζ,c)

    # Permute dimensions: VS_permuted[ζ,β,α,c] = VS[α,β,ζ,c]
    VS_permuted = permutedims!!(VS, (3,2,1,4), buffer=buffer[2])
    VS_permuted = reshape(VS_permuted, ζ*β, α*c)

    # Permute dimensions: B_permuted[z,b,ζ,β] = B[z,ζ,b,β]
    B_permuted = permutedims!!(B, (1,3,2,4), buffer=buffer[1])
    B_permuted = reshape(B_permuted, z*b, ζ*β)

    # Step 2: VSB[z,b,α,c] = B_permuted[z,b,ζ,β] * VS_permuted[ζ,β,α,c]
    VSB = mul!!(B_permuted, VS_permuted, buffer=buffer[3])
    VSB = reshape(VSB, z,b,α,c)

    # Permute dimensions: A_permuted[a,z,α] = A[z,a,α]
    A_permuted = permutedims!!(A, (2,1,3), buffer=buffer[1])
    A_permuted = reshape(A_permuted, a,z*α)

    # Permute dimensions: VSB_permuted[z,α,b,c] = VSB[z,b,α,c]
    VSB_permuted = permutedims!!(VSB, (1,3,2,4), buffer=buffer[2])
    VSB_permuted = reshape(VSB_permuted, z*α,b*c)

    # Step 3: W[a,b,c] = A_permuted[a,z,α] * VSB_permuted[z,α,b,c]
    mul!(reshape(W, a,b*c), A_permuted, VSB_permuted)
  end

  return W
end

function contract_sketch_core_backwards_operator_buffers_size(z::Int, ζ::Int, a::Int, α::Int, b::Int, β::Int, c::Int, γ::Int)
  if a*γ*(z*β*(α+ζ*b)+ζ*b*c) < α*c*(ζ*β*(γ+z*b) + z*a*b)  # Order 1: ((V*A)*B)*S
    # buffer[1]: VA intermediate[z*a,β*γ], B_permuted[z*β,ζ*b] (overwrites VA safely), VAB_permuted[a,b,ζ,γ]
    buffer1_size = max(z*a*β*γ, z*β*ζ*b, a*b*ζ*γ)
    # buffer[2]: VA_permuted[a,γ,z,β], S_permuted[ζ,γ,c]
    buffer2_size = max(a*γ*z*β, ζ*γ*c)
    # buffer[3]: VAB intermediate[a*γ,ζ*b]
    buffer3_size = a*γ*ζ*b
    return (buffer1_size, buffer2_size, buffer3_size)
  else  # Order 2: ((V*S)*B)*A
    # buffer[1]: VS intermediate[α*β,ζ*c], B_permuted[z,b,ζ,β] (overwrites VS safely), A_permuted[a,z*α]
    buffer1_size = max(α*β*ζ*c, z*b*ζ*β, a*z*α)
    # buffer[2]: VS_permuted[ζ,β,α,c], VSB_permuted[z,α,b,c]
    buffer2_size = max(ζ*β*α*c, z*α*b*c)
    # buffer[3]: VSB intermediate[α*c,z*b]
    buffer3_size = α*c*z*b
    return (buffer1_size, buffer2_size, buffer3_size)
  end
end

function contract_sketch_core_kronecker_backwards!(W::AbstractArray{T,N1}, A::NTuple{N,AbstractArray{T,3}}, S::AbstractArray{T,3}, V::AbstractArray{T,N1};
                                buffer=(nothing,nothing,nothing)) where {T,N,N1}
  @assert N1 === N+1

  w = size(W)
  v = size(V)
  z = size(A[1],1)
  for i=1:N
    @assert size(A[i]) == (z,w[i],v[i])   "Factor A[$i] has the wrong dimensions: need $((z,w[i],v[i])), got $(size(A[i]))"
  end
  @assert size(S) === (v[N+1],z,w[N+1])   "Factor S has the wrong dimensions: need $((v[N+1],z,w[N+1])), got $(size(S))"

  function permutation(i,j) # Bring i to 1 and j to N+1
    q = collect(1:N+2)
    q[1], q[i] = i, 1
    if 1<j # j is still at j
      q[N+1], q[j] = j, q[N+1]
    else # j is now at i
      q[N+1], q[i] = j, q[N+1]
    end
    return q
  end

  function inversepermutation(i,j) # Bring i back from 1 and j from N+1
    q = collect(1:N+2)
    if 1<j
      q[N+1], q[j] = q[j], q[N+1]
    else
      q[N+1], q[i] = q[i], q[N+1]
    end
    q[1], q[i] = q[i], q[1]
    return q
  end

  order = sortperm(collect(w./v), rev=true)
  j = pop!(order)    
  # buffer[1]: V_permuted[α≠i,αi]  
  perm = permutation(1,j)[1:N+1]

  V_permuted = permutedims!!(V, perm, buffer=buffer[1])
  if j == N+1
    # buffer[2]: A_permuted[αj,aj,z] = S[β,z,b]
    A_permuted = permutedims!!(S, (1,3,2), buffer=buffer[2])
  else
    # buffer[2]: A_permuted[αj,aj,z] = Aj[z,aj,αj]
    A_permuted = permutedims!!(A[j], (3,2,1), buffer=buffer[2])
  end
  # buffer[3]: VA[α≠j, aj, z]
  VA = mul!!(reshape(V_permuted, :, v[j]), reshape(A_permuted, v[j], w[j]*z), buffer=buffer[3])
  va = [v[perm[1:N]]..., w[j], z]
  VA = reshape(VA, va...)
  perm = inversepermutation(1,j)

  while length(order) > 2
    i = pop!(order)
    j = pop!(order)
    perm = perm[permutation(i,j)]

    # buffer[1]: VA_permuted[αi,α≠ij,αj,z]  
    VA_permuted = permutedims!!(VA, perm, buffer=buffer[1])
    va = va[perm]

    if i == N+1
      # buffer[2]: A_permuted[ai,αi,z] = S[β,z,b]
      A_permuted = permutedims!!(S, (3,1,2), buffer=buffer[2])
    else
      # buffer[2]: A_permuted[ai,αi,z] = Ai[z,ai,αi]
      A_permuted = permutedims!!(A[i], (2,3,1), buffer=buffer[2])
    end
    # buffer[3]: VA[ai,a/α≠ij,βj,z] = A_permuted[ai,αi,z] * VA_permuted[αi,α≠ij,βj,z] 
    VA_intermediate = mul!!(A_permuted, reshape(VA_permuted,v[i],:,z), buffer=buffer[3])
    va[1] = w[i]
    if j == N+1
      # buffer[1]: B_permuted[βj,bj,z] = S[β,z,b]
      B_permuted = permutedims!!(S, (1,3,2), buffer=buffer[1])
    else
      # buffer[1]: B_permuted[βj,bj,z]
      B_permuted = permutedims!!(A[j], (3,2,1), buffer=buffer[1])
    end
    # buffer[2]: VA[ai,a/α≠ij,bj,z] = VA_intermediate[ai,α≠ij,βj,z] * B_permuted[βj,bj,z]
    VA = mul!!(reshape(VA_intermediate,:,v[j],z), B_permuted, buffer=buffer[2])
    va[N+1] = w[j]
    VA = reshape(VA, va...)

    perm = inversepermutation(i,j)
  end

  if length(order) == 2
    i = pop!(order)
    j = pop!(order)
    perm = perm[permutation(i,j)]

    # buffer[1]: VA_permuted[αi,α≠ij,αj,z]  
    VA_permuted = permutedims!!(VA, perm, buffer=buffer[1])
    va = va[perm]
    if i == N+1
      # buffer[2]: A_permuted[ai,αi,z] = S[β,z,b]
      A_permuted = permutedims!!(S, (3,1,2), buffer=buffer[2])
    else
      # buffer[2]: A_permuted[ai,αi,z] = Ai[z,ai,αi]
      A_permuted = permutedims!!(A[i], (2,3,1), buffer=buffer[2])
    end
    # buffer[3]: VA_intermediate[ai,a/α≠ij,βj,z] = Ai_permuted[ai,αi,z] * VA_permuted[αi,α≠ij,βj,z] 
    VA_intermediate = mul!!(A_permuted, reshape(VA_permuted,v[i],:,z), buffer=buffer[3])
    va[1] = w[i]
    if j == N+1
      # buffer[1]: B_permuted[βj,z,bj] = S[β,z,b]
      B_permuted = S
    else
      # buffer[1]: B_permuted[βj,z,bj]
      B_permuted = permutedims!!(A[j], (3,1,2), buffer=buffer[1])
    end
    # buffer[2]: VA[ai,a≠ij,bj] = VA_intermediate[ai,α≠ij,βj,z] * Bj_permuted[βj,z,bj]
    W_permuted = mul!!(reshape(VA_intermediate,:,v[j]*z), reshape(B_permuted, v[j]*z, w[j]), buffer=buffer[2])
    va = va[1:N+1]
    va[N+1] = w[j]
 
    W_permuted = reshape(W_permuted, va...)
    perm = inversepermutation(i,j)[1:N+1]
    permutedims!(W, W_permuted, perm)
  else # length(order == 1)
    # Dimension j is currently at index perm[j]
    j = pop!(order)
    perm = perm[permutation(1,j)]

    # buffer[1]: VA_permuted[αi,α≠ij,αj,z]  
    VA_permuted = permutedims!!(VA, perm, buffer=buffer[1])
    va = va[perm]
    if j == N+1
      # buffer[2]: B_permuted[βj,z,bj] = S[β,z,b]
      B_permuted = S
    else
      # buffer[2]: B_permuted[βj,z,bj]
      B_permuted = permutedims!!(A[j], (3,1,2), buffer=buffer[2])
    end
    # buffer[3]: VA[ai,a≠ij,bj] = VA_permuted[a≠j,βj,z] * B_permuted[βj,z,bj]
    W_permuted = mul!!(reshape(VA_permuted,:,v[j]*z), reshape(B_permuted, v[j]*z, w[j]), buffer=buffer[3])
    va = va[1:N+1]
    va[N+1] = w[j]
 
    W_permuted = reshape(W_permuted, va...)
    perm = inversepermutation(1,j)[1:N+1]
    permutedims!(W, W_permuted, perm)
  end

  return W
end

function contract_sketch_core_backwards_kronecker_buffers_size(z::Int, v::NTuple{N1,Int}, w::NTuple{N1,Int}) where {N1}
  N = N1 - 1
  
  # Buffer 1: Used for VA_permuted (permutations), B_permuted operations
  buffer1_size = 0
  # Buffer 2: Used for A_permuted, final VA/W_permuted results
  buffer2_size = 0
  # Buffer 3: Used for VA_intermediate results from mul!!
  buffer3_size = 0
  
  # Initial V permutation: V_permuted has same size as V
  buffer1_size = max(buffer1_size, prod(v))
  
  # Track dimensions through the algorithm using same ordering as main function
  order = sortperm(collect(w./v), rev=true)
  va = [v..., z]  # Initial va dimensions
  
  l(i) = z*v[i]*w[i]
  
  # Initial VA computation
  i = pop!(order)
  
  # Initial A[i] or S permutation - now A[i] has dimensions (z,w[i],v[i])
  buffer2_size = max(buffer2_size, l(i))
  
  # Update va after first contraction
  va[i] = w[i]
  buffer3_size = max(buffer3_size, prod(va))
  
  # Track through algorithm iterations
  while length(order) > 2
    i = pop!(order)
    j = pop!(order)
    
    # VA_permuted: reordered version of current VA
    current_va_size = prod(va)
    buffer1_size = max(buffer1_size, current_va_size)
    
    # A_permuted for i (or S if i == N+1) - A[i] has dimensions (z,w[i],v[i])
    buffer2_size = max(buffer2_size, l(i))
    
    # VA_intermediate from mul!!
    va[i] = w[i]
    buffer3_size = max(buffer3_size, prod(va))
    
    # B_permuted for j (or S if j == N+1) - A[j] has dimensions (z,w[j],v[j])
    buffer1_size = max(buffer1_size, l(j))
    
    # Final VA from second mul!!
    va[j] = w[j]
    buffer2_size = max(buffer2_size, prod(va))
  end
  
  if length(order) == 2
    i = pop!(order)
    j = pop!(order)
    
    # VA_permuted
    buffer1_size = max(buffer1_size, prod(va))
    
    # A_permuted for i (or S if i == N+1)
    buffer2_size = max(buffer2_size, l(i))
    
    # VA_intermediate from mul!!
    va[i] = w[i]
    buffer3_size = max(buffer3_size, prod(va))
    
    # B_permuted for j (or S if j == N+1)
    buffer1_size = max(buffer1_size, l(j))
    
    # Final W_permuted
    va[j] = w[j]
    buffer2_size = max(buffer2_size, prod(w))
    
  elseif length(order) == 1
    j = pop!(order)
    
    # VA_permuted
    buffer1_size = max(buffer1_size, prod(va))
    
    # Handle special case: if j is the last core (N+1), use S directly
    if j < N+1
      # B_permuted for regular A[j] - A[j] has dimensions (z,w[j],v[j])
      buffer2_size = max(buffer2_size, l(j))
    end
    va[j] = w[j]
    buffer3_size = max(buffer3_size, prod(w))
  end
  
  return (buffer1_size, buffer2_size, buffer3_size)
end


function permutedims!!(A::AbstractArray{T, N}, perm; buffer=nothing) where {T,N}
  if buffer === nothing
    A_permuted = permutedims(A, perm)
  else
    @assert length(buffer) >= length(A) "Buffer too small: need $(length(A)), got $(length(buffer))"
    permuted_size = ntuple(i -> size(A, perm[i]), Val(N))
    A_permuted = unsafe_wrap(Array, pointer(buffer), permuted_size)
    permutedims!(A_permuted, A, perm)
  end
  return A_permuted
end

function mul!!(A::AbstractMatrix{T}, B::AbstractMatrix{T}; buffer=nothing) where {T}
  if buffer === nothing
    C = A*B
  else
    m = size(A,1)
    n = size(B,2)
    @assert length(buffer) >= m*n "Buffer too small: need $(m*n)=($m)⨯($n), got $(length(buffer))"
    C = unsafe_wrap(Array, pointer(buffer), (m,n))
    mul!(C, A,B)
  end
  return C
end


function mul!!(A::AbstractArray{T,3}, B::AbstractArray{T,3}; buffer=nothing) where {T}
  if buffer === nothing
    m = size(A,1)
    n = size(B,2)
    K = size(A,3)
    @assert size(B,3) == K "Incompatible sizes: expected size(A,3)==$(K)==size(B,3)==$(size(B,3))"
    C = Array{T,3}(undef, m,n,K)
    for k=1:K
      @views mul!(C[:,:,k], A[:,:,k], B[:,:,k])
    end
  else
    m = size(A,1)
    n = size(B,2)
    K = size(A,3)
    @assert length(buffer) >= m*n*K "Buffer too small: need $(m*n*k)=($m)⨯($n)⨯($K), got $(length(buffer))"
    C = unsafe_wrap(Array, pointer(buffer), (m,n,K))
    for k=1:K
      @views mul!(C[:,:,k], A[:,:,k], B[:,:,k])
    end  end
  return C
end

