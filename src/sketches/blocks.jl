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
