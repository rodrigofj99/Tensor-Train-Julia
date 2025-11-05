using TensorTrains
using LinearAlgebra
using StaticArrays
using Random
using TensorOperations

"""
    generate_sketch_blocks(rng, ::Type{T}, dim, left_rank, right_rank, p, orthogonal) where T

Generate random sketch tensor blocks for tensor train sketching, assuming forward sketch direction

# Arguments
- `rng`: Random number generator
- `T::Type`: Element type for the sketch tensor
- `dim::Int`: Physical dimension at this mode
- `left_rank::Int`: Left block rank
- `right_rank::Int`: Right block rank (will be updated if identity optimization applies)
- `p::Int`: Number of sketch blocks
- `orthogonal::Bool`: Whether to use orthogonal sketches (QR decomposition)

# Returns
- `B_sketch::Array{T,4}`: Sketch tensor of size (dim, left_rank, right_rank, p)
- `right_rank::Int`: Updated right block rank (may change with identity optimization)

# Algorithm
Three cases:
1. Identity optimization: When right_rank ≥ dim*left_rank, uses identity matrix structure
2. Orthogonal QR: When orthogonal=true and ranks don't satisfy identity condition, uses QR factorization
3. Random normalized: Otherwise, uses normalized random tensor normalized by 1/sqrt(dim*left_rank)
"""
function generate_sketch_blocks(rng, ::Type{T}, dim, left_rank, right_rank, p, orthogonal) where T
  if orthogonal && (right_rank < dim * left_rank)
    block = zeros(T, dim*left_rank, right_rank, p)
    # QR orthogonalization
    @inbounds for j=1:p
        q,_ = qr!(randn(rng, T, dim*left_rank, right_rank), ColumnNorm())
        block[:,:,j] .= Array(q)
    end
  elseif orthogonal
    # Identity 'sketch'
    right_rank = dim * left_rank

    block = zeros(T, dim*left_rank, right_rank, p)
    for j=1:p, i in 1:right_rank
      block[i,i,j] = 1
    end
  else
    # Simple normalization
    block = randn(rng, T, dim, left_rank, right_rank, p)
    block .*= 1/sqrt(dim * left_rank)
  end
  return reshape(block, dim, left_rank, right_rank, p), right_rank
end

"""
    tt_recursive_sketch([T=Float64,] [H::TToperator,] A::TTvector, rks;       orthogonal=true, reverse=true, seed=1234) -> (W, sketch_rks)
    tt_recursive_sketch([T=Float64,] [H::TToperator,] A::TTvector, rmax::Int; orthogonal=true, reverse=true, seed=1234) -> (W, sketch_rks)

Compute a recursive sketch of a TTvector A, or of the result of applying TToperator H to TTvector A,
without explicitly forming the product, using random projections.

Generates sketch matrices W[k] for each TT core by recursively contracting with random
orthogonal (or normalized) tensors. The sketch can be used for efficient randomized
algorithms like ttrand_rounding.

# Arguments
- `T::Type{<:Number}`: Element type for random sketch tensors (optional, default: Float64)
- `H::TToperator{TH,N}`: TToperator to apply (optional)
- `A::TTvector{TA,N}`: Input TTvector to sketch
- `rks::Vector{Int}` or `rmax::Int`: Target sketch ranks (length N+1 with boundary conditions rks[1]=1 or rks[N+1]=1) or maximum sketch rank

# Keyword Arguments
- `orthogonal::Bool=true`: Generate orthogonal random tensors (via QR) for better numerical stability
- `reverse::Bool=true`: Sweep direction (true: right-to-left, false: left-to-right)
- `seed::Int=1234`: Random seed for reproducibility

# Returns
- `W::Vector{Matrix}`: Sketch matrices where W[k] has size (A.ttv_rks[k], sketch_rks[k])
- `sketch_rks::Vector{Int}`: Actual sketch ranks achieved (may differ from input rks due to heuristics)

# Algorithm
Uses a block sketching strategy exploiting the Tensor-Train structure, 
with oversampling to ensure probabilistic guarantees. 
When orthogonal=true, applies QR factorization with column
pivoting to each random block.

# References
- Algorithm based on randomized tensor decomposition methods
- See ttrand_rounding for usage example
"""
function tt_recursive_sketch(::Type{T}, A::TTvector{TA,N}, rks; orthogonal=true, reverse=true, seed=1234) where {T<:Number,TA<:Number,N}
  rng = Random.default_rng()
  Random.seed!(rng, seed)
  
  dims = A.ttv_dims
  TW = typeof(one(T)*one(TA))
  W = Vector{Array{TW,3}}(undef, N+1)

  if reverse
    @assert rks[N+1] == 1 && A.ttv_rks[N+1] == 1
    block_rks = ones(Int, N+1)

    block_rks[1:N] .= N # Heuristic, to be played with. Uniform may not be best!

    p = @. ceil(Int, 2*rks/block_rks) # Heuristic
    for i=2:N
      p[i] = max(p[i], p[i-1]) # Make sure the number of blocks is strictly increasing
    end
    p[N+1] = 1

    W[N+1] = ones(TW,1,1,1)
    @inbounds for k in N:-1:1
      B_sketch, block_rks[k] = generate_sketch_blocks(rng, T, dims[k], block_rks[k+1], block_rks[k], p[k], orthogonal)
      B_sketch = permutedims(B_sketch, (1,3,2,4))  # Swap to (dim, block_rks[k], block_rks[k+1], p)

      W[k] = zeros(TW, A.ttv_rks[k], block_rks[k], p[k])
      for j=1:p[k]
        W_next_j = view(W[k+1],:,:,(k<N ? j : 1))
        B_j = view(B_sketch,:,:,:,j)
        W_k_j = view(W[k],:,:,j)
        @tensoropt((a,b,α,β,z), W_k_j[a,b] = A.ttv_vec[k][z,a,α]*B_j[z,b,β]*W_next_j[α,β]) #size R^A_{k} × R^B_{k}
      end
    end
  else
    @assert rks[1] == 1 && A.ttv_rks[1] == 1
    block_rks = ones(Int, N+1)
    block_rks[2:N+1] .= N

    p = ceil.(Int, @. 2*rks/block_rks) # Heuristic
    for i=N:-1:2
      p[i] = max(p[i], p[i+1]) # Make sure the number of blocks is strictly decreasing
    end
    p[1] = 1

    W[1] = ones(TW,1,1,1)
    @inbounds for k in 1:N
      B_sketch, block_rks[k+1] = generate_sketch_blocks(rng, T, dims[k], block_rks[k], block_rks[k+1], p[k+1], orthogonal)

      W[k+1] = zeros(TW, A.ttv_rks[k+1], block_rks[k+1], p[k+1])
      for j=1:p[k+1]
        W_k_j = view(W[k],:,:,(k>1 ? j : 1))
        B_j = view(B_sketch,:,:,:,j)
        W_next_j = view(W[k+1],:,:,j)
        @tensoropt((a,b,α,β,z), W_next_j[a,b] = A.ttv_vec[k][z,α,a]*B_j[z,β,b]*W_k_j[α,β]) #size R^A_{k} × R^B_{k}
      end 
    end
  end

  return [reshape(W[k], A.ttv_rks[k], block_rks[k]*p[k]) for k=1:N+1], block_rks.*p
end

function tt_recursive_sketch(::Type{T}, A::TTvector{TA,N}, rmax::Int; orthogonal=true, reverse=true, seed=1234) where {T<:Number,TA<:Number,N}
  rks = rmax*ones(Int,N+1)
  rks[(reverse ? N+1 : 1)] = 1
  return tt_recursive_sketch(T,A,rks; orthogonal=orthogonal, reverse=reverse, seed=seed)
end

function tt_recursive_sketch(A::TTvector{T,N},rks_or_rmax; orthogonal=true, reverse=true, seed=1234) where {T<:Number,N}
  return tt_recursive_sketch(Float64,A,rks_or_rmax; orthogonal=orthogonal, reverse=reverse, seed=seed)
end

function tt_recursive_sketch(::Type{T}, H::TToperator{TH,N}, A::TTvector{TA,N}, rks; orthogonal=true, reverse=true, seed=1234) where {T<:Number,TA<:Number,TH<:Number,N}
  rng = Random.default_rng()
  Random.seed!(rng, seed)

  dims = A.ttv_dims
  TW = typeof(one(T)*one(TA)*one(TH))
  W = Vector{Array{TW,4}}(undef, N+1)

  if reverse
    @assert rks[N+1] == 1 && A.ttv_rks[N+1] == 1 && H.tto_rks[N+1] == 1
    block_rks = ones(Int, N+1)

    block_rks[1:N] .= N # Heuristic, to be played with. Uniform may not be best!

    p = @. ceil(Int, 2*rks/block_rks) # Heuristic
    for i=2:N
      p[i] = max(p[i], p[i-1]) # Make sure the number of blocks is strictly increasing
    end
    p[N+1] = 1

    W[N+1] = ones(TW,1,1,1,1)
    @inbounds for k in N:-1:1
      B_sketch, block_rks[k] = generate_sketch_blocks(rng, T, dims[k], block_rks[k+1], block_rks[k], p[k], orthogonal)
      B_sketch = permutedims(B_sketch, (1,3,2,4))  # Swap to (dim, block_rks[k], block_rks[k+1], p)

      W[k] = zeros(TW, A.ttv_rks[k], H.tto_rks[k], block_rks[k], p[k])
      for j=1:p[k]
        W_next_j = view(W[k+1],:,:,:,(k<N ? j : 1))
        B_j = view(B_sketch,:,:,:,j)
        W_k_j = view(W[k],:,:,:,j)
        @tensoropt((a,b,h,α,β,η,y,z), W_k_j[a,h,b] = A.ttv_vec[k][z,a,α]*H.tto_vec[k][y,z,h,η]*B_j[y,b,β]*W_next_j[α,η,β])
      end
    end
  else
    @assert rks[1] == 1 && A.ttv_rks[1] == 1 && H.tto_rks[1] == 1
    block_rks = ones(Int, N+1)
    block_rks[2:N+1] .= N

    p = ceil.(Int, @. 2*rks/block_rks) # Heuristic
    for i=N:-1:2
      p[i] = max(p[i], p[i+1]) # Make sure the number of blocks is strictly decreasing
    end
    p[1] = 1

    W[1] = ones(TW,1,1,1,1)
    @inbounds for k in 1:N
      B_sketch, block_rks[k+1] = generate_sketch_blocks(rng, T, dims[k], block_rks[k], block_rks[k+1], p[k+1], orthogonal)

      W[k+1] = zeros(TW, A.ttv_rks[k+1], H.tto_rks[k+1], block_rks[k+1], p[k+1])
      for j=1:p[k+1]
        W_k_j = view(W[k],:,:,:,(k>1 ? j : 1))
        B_j = view(B_sketch,:,:,:,j)
        W_next_j = view(W[k+1],:,:,:,j)
        @tensoropt((a,b,h,α,β,η,y,z), W_next_j[a,h,b] = A.ttv_vec[k][z,α,a]*H.tto_vec[k][y,z,η,h]*B_j[y,β,b]*W_k_j[α,η,β])
      end 
    end
  end
  return [reshape(W[k], A.ttv_rks[k], H.tto_rks[k], block_rks[k]*p[k]) for k=1:N+1], block_rks.*p
end

function tt_recursive_sketch(::Type{T}, H::TToperator{TH,N}, A::TTvector{TA,N}, rmax::Int; orthogonal=true, reverse=true, seed=1234) where {T<:Number,TA<:Number,TH<:Number,N}
  d = A.N
  rks = rmax*ones(Int,d+1)
  rks[(reverse ? d+1 : 1)] = 1
  return tt_recursive_sketch(T,H,A,rks; orthogonal=orthogonal, reverse=reverse, seed=seed)
end

function tt_recursive_sketch(H::TToperator{TH,N}, A::TTvector{TA,N}, rks_or_rmax; orthogonal=true, reverse=true, seed=1234) where {TA<:Number,TH<:Number,N}
  return tt_recursive_sketch(Float64, H, A, rks_or_rmax; orthogonal=orthogonal, reverse=reverse, seed=seed)
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
      @tensoropt((a,b,α,β,z), W[k][a,b] = A.ttv_vec[k][z,a,α]*B.ttv_vec[k][z,b,β]*W[k+1][α,β]) #size R^A_{k} × R^B_{k}
    end
  else
    W[1] = ones(T,1,1)
    @inbounds for k in 1:L
      @tensoropt((a,b,α,β,z), W[k+1][a,b] = A.ttv_vec[k][z,α,a]*B.ttv_vec[k][z,β,b]*W[k][α,β]) #size R^A_{k} × R^B_{k}
    end
  end
  return W
end

"""
    stta_sketch(x::TTvector, L::TTvector, R::TTvector) -> (Ω, Ψ)
    stta_sketch(x::TTvector, rks::Vector{Int}; seed_left=1234, seed_right=5678, orthogonal=true) -> (Ω, Ψ)

Compute left and right sketches for Streaming Tensor Train Approximation (STTA).

Generates two-sided random sketches of a TTvector using either explicit random TTvectors
or recursive sketching with target ranks. For the recursive version, automatically applies
50% oversampling to left ranks for optimal STTA performance.

# Arguments
- `x::TTvector{T,N}`: Input TTvector to sketch
- `L::TTvector{T,N}`: Left random TTvector (for explicit sketch version)
- `R::TTvector{T,N}`: Right random TTvector (for explicit sketch version)
- `rks::Vector{Int}`: Target approximation ranks (for recursive sketch version)

# Keyword Arguments (recursive sketch version)
- `seed_left::Int=1234`: Random seed for left sketch
- `seed_right::Int=5678`: Random seed for right sketch
- `orthogonal::Bool=true`: Use orthogonal random sketches

# Returns
- `Ω::Vector{Matrix}`: Overlap matrices where Ω[k] = L'[k+1] * R[k+1]
- `Ψ::Vector{Array{T,3}}`: Sketched cores where Ψ[k][i,α,β] = x[i] * L[α] * R[β]

# Algorithm
Based on:
Kressner, Vandereycken & Voorhaar (2022), "Streaming Tensor Train Approximation",
SIAM J. Sci. Comput., 45(5), pp. A2610–A2629, https://arxiv.org/abs/2208.02600

For optimal STTA performance, left sketch ranks are automatically set to 50% larger
than the target ranks: l_rks[2:N] = ceil.(Int, 1.5 .* rks[2:N]).

The sketch provides a compressed representation that can be efficiently used for
approximation via the stta function.
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
                     seed_left::Int=1234, seed_right::Int=5678, orthogonal::Bool=true) where {T,N}
  # For optimal STTA performance, left ranks should be 50% larger than target ranks
  r_rks = rks  # Right sketch uses target ranks
  l_rks = ones(Int, N+1)
  for k=2:N
    l_rks[k] = ceil(Int, 1.5*r_rks[k]) # Left sketch with 50% oversampling
  end
  
  # Generate left sketch (forward direction)
  W_left, sketch_l_rks = tt_recursive_sketch(T, x, l_rks; orthogonal=orthogonal, reverse=false, seed=seed_left)

  # Generate right sketch (reverse direction)
  W_right, sketch_r_rks = tt_recursive_sketch(T, x, r_rks; orthogonal=orthogonal, reverse=true, seed=seed_right)

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

