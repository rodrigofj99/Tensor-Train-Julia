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
  # Ψ cores carry the same (L, I, R) layout convention as TTvector cores.
  Ψ = [zeros(T, L.ttv_rks[i], x.ttv_dims[i], R.ttv_rks[i+1]) for i in 1:N]
  Ω = [zeros(T, L.ttv_rks[i+1], R.ttv_rks[i+1]) for i in 1:N-1]
  left_contractions = partial_contraction(x,L;reverse=false)
  right_contractions = partial_contraction(x,R;reverse=true)
  for k in eachindex(Ω)
    @tensor Ω[k][a,b] = left_contractions[k+1][z,a]*right_contractions[k+1][z,b]
  end
  for k in eachindex(Ψ)
    @tensor (Ψ[k][α,i,β] = (x.ttv_vec[k][y,i,z]*left_contractions[k][y,α])*right_contractions[k+1][z,β])
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

    # Compute Ω and Ψ from the sketch matrices. Ψ cores adopt (L, I, R) layout.
    Ψ = [zeros(T, sketch_l_rks[i], x.ttv_dims[i], sketch_r_rks[i+1]) for i in 1:N]
    Ω = [zeros(T, sketch_l_rks[i+1], sketch_r_rks[i+1]) for i in 1:N-1]

    # Contract to form Ω[k] = W_left[k+1]' * W_right[k+1]
    for k in 1:N-1
      @tensor Ω[k][a,b] = W_left[k+1][α,a] * W_right[k+1][α,b]
    end

    # Contract to form Ψ[k]. Vector core (L, I, R) = (α, i, β).
    for k in 1:N
      @tensor Ψ[k][a,i,b] = x.ttv_vec[k][α,i,β] * W_left[k][α,a] * W_right[k+1][β,b]
    end

    return Ω, Ψ
  end
end
