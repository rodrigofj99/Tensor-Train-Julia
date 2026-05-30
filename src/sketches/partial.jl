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
  # Cores have layout (L, I, R). Each ttv_vec[k] is indexed as [left_rank, phys, right_rank].
  if reverse
    W[L+1] = ones(T,1,1)
    @inbounds for k in L:-1:1
      # W[k][a, b] = sum_{z, α, β} A[a, z, α] × W[k+1][α, β] × B[b, z, β]. Two gemms.
      cA, cB = A.ttv_vec[k], B.ttv_vec[k]
      a, zd, αd = size(cA); b, _, βd = size(cB)
      # Step 1: AW[a, z, β] = A[a*z, α] × W[k+1][α, β].
      AW = reshape(cA, a*zd, αd) * W[k+1]
      # Step 2: W[k][a, b] = sum_{z, β} reshape(AW, a, z*β) × transpose(reshape(B, b, z*β)).
      mul!(W[k],
           reshape(AW, a, zd*βd),
           transpose(reshape(cB, b, zd*βd)))
    end
  else
    W[1] = ones(T,1,1)
    @inbounds for k in 1:L
      # W[k+1][a, b] = sum_{z, α, β} A[α, z, a] × W[k][α, β] × B[β, z, b]. Two gemms.
      cA, cB = A.ttv_vec[k], B.ttv_vec[k]
      αd, zd, a = size(cA); βd, _, b = size(cB)
      # Step 1: WB[α, z, b] = W[k][α, β] × B[β, z*b].
      WB = W[k] * reshape(cB, βd, zd*b)
      # Step 2: W[k+1][a, b] = transpose(reshape(A, α*z, a)) × reshape(WB, α*z, b).
      mul!(W[k+1],
           transpose(reshape(cA, αd*zd, a)),
           reshape(WB, αd*zd, b))
    end
  end
  return W
end
