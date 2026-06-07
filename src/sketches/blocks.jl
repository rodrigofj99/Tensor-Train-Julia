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
# Deterministic per-block seed. A block's Gaussian draw depends only on
# (base_seed, bond_index, block_index, reverse, group) — NOT on the call's RNG history — so a
# recursive sketch and its adaptive per-bond extensions are content-addressable and can be
# cached/extended reproducibly across calls. `reverse` is mixed in because the forward and
# reverse sweeps draw differently-shaped blocks at the same bond. `group` gives disjoint block
# namespaces so independent sketch groups (e.g. an `block_rks` initial sketch and a
# `block_rks_inc` extension) draw independent blocks even at the same (bond, block_index).
@inline _block_seed(base_seed::Integer, bond_index::Integer, block_index::Integer, reverse::Bool, group::Integer) =
    hash((UInt64(0x53_4b_45_54_43_48_00), Int(base_seed), Int(bond_index), Int(block_index), reverse, Int(group)))

function generate_sketch_blocks(base_seed::Integer, bond_index::Integer, block_offset::Integer,
                                ::Type{T}, left_rank, dim, right_rank, p, orthogonal;
                                reverse::Bool=true, group::Integer=0, buffer=nothing, timer::TimerOutput = TimerOutput()) where T
  @assert (!orthogonal) || right_rank <= dim * left_rank
  use_identity = orthogonal && (right_rank == dim * left_rank)
  m = left_rank * dim

  @timeit timer "Block allocation" begin
    if buffer === nothing
      block = Array{T,3}(undef, m, right_rank, p)
    else
      block_size = m*right_rank*p
      @assert length(buffer) >= block_size "Buffer too small: need $block_size, got $(length(buffer))"
      block = unsafe_wrap(Array, pointer(buffer), (m, right_rank, p))
    end
  end

  # Per-block Gaussian fill: block j (global index block_offset+j) gets its own seeded rng.
  if !use_identity
    @timeit timer "random number generator" begin
      @inbounds for j in 1:p
        rng_j = Random.Xoshiro(_block_seed(base_seed, bond_index, block_offset + j, reverse, group))
        randn!(rng_j, @view block[:, :, j])
      end
    end
  end

  if use_identity
    @timeit timer "identity block creation" begin
      fill!(block, T(0))
      for j=1:p, i in 1:right_rank
        block[i,i,j] = 1
      end
    end
  elseif orthogonal && right_rank == 1
    # Fast path: each block is a single column of length m, orthonormal means
    # unit-norm. Skip the factorisation entirely — Gaussian + per-column
    # normalise is identical to Stewart for right_rank=1. Spectral scaling sqrt(m/1).
    @timeit timer "normalize+scale" begin
      scale_factor = sqrt(T(m))
      @inbounds for j in 1:p
        col = @view block[:, 1, j]
        col .*= scale_factor / norm(col)
      end
    end
  elseif orthogonal
    # Stewart's algorithm: build each block's Q from random Householders via
    # LAPACK.orgqr! (blocked WY assembly), ~1.5× faster than qr! + Matrix(F.Q).
    @timeit timer "stewart_householders" begin
      tau = Vector{T}(undef, right_rank)
      @inbounds for j in 1:p
        _stewart_block!(@view(block[:, :, j]), tau)
      end
    end
    @timeit timer "normalization" block .*= sqrt(T(m) / T(right_rank))
  else # Non-orthogonal Gaussian sketch
    @timeit timer "normalization" block .*= 1/sqrt(T(right_rank))
  end
  return reshape(block, left_rank, dim, right_rank, p)
end

# In-place Stewart for a single block: A is (m, n), random Gaussian on input.
# Replaces A with Q (orthonormal columns) via random Householders + LAPACK.orgqr!.
function _stewart_block!(A::AbstractMatrix{T}, tau::AbstractVector{T}) where T<:AbstractFloat
    m, n = size(A)
    @inbounds for k in 1:n
        # Inline larfg on column k: turn A[k:m, k] into a Householder reflector
        # H = I − τ·v·v^T with v[1]=1 implicit (stored as scaling), v[2:] in A[k+1:m, k].
        # On exit A[k, k] holds β = ±‖A[k:m, k]‖ (sign opposite α), A[k+1:m, k] the
        # rescaled tail, and tau[k] the scalar τ.
        a = A[k, k]
        xnorm2 = zero(T)
        @simd for i in (k+1):m
            xnorm2 += A[i, k] * A[i, k]
        end
        if xnorm2 == zero(T) && a >= zero(T)
            tau[k] = zero(T)
        else
            β = -copysign(sqrt(a*a + xnorm2), a)
            tau[k] = (β - a) / β
            scale = one(T) / (a - β)
            @simd for i in (k+1):m
                A[i, k] *= scale
            end
            A[k, k] = β
        end
    end
    # Materialize Q from the reflectors via LAPACK's blocked routine
    LAPACK.orgqr!(A, tau)
    return A
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
