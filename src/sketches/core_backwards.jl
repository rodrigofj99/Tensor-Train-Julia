"""
    contract_sketch_core_backwards!(W, A, S, V; buffer=(nothing,nothing), timer=TimerOutput())
    contract_sketch_core_backwards!(W, A, B, S, V; buffer=(nothing,nothing,nothing), timer=TimerOutput())

Efficient contraction kernel for backward tensor train sketching:
W[a,b] = A[z,a,α] * S[β,z,b] * V[α,β]
or
W[a,b,c] = A[z,a,α] * B[ζ,z,b,β] * S[γ,ζ,c] * V[α,β,γ]

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
- `B::AbstractArray{T,4}`: Operator tensor [ζ,z,b,β] (backward TTO format)
- `S::AbstractArray{T,3}`: Sketch block tensor [γ,ζ,c] (sketch block format)
- `V::AbstractArray{T,3}`: Connection array [α,β,γ] (previous sketch weights)

# Optional arguments
- `buffer::NTuple{Union{Nothing,AbstractVector}}`: Optional preallocated buffers
- `timer::TimerOutput`: Optional timer for performance profiling

"""
function contract_sketch_core_backwards!(W::AbstractMatrix{T}, A::AbstractArray{T,3}, S::AbstractArray{T,3}, V::AbstractMatrix{T};
                                buffer=(nothing,nothing)) where T

  # A has the new TT layout (L, I, R) = (a, z, α) for the backwards sweep.
  a,b = size(W)
  α,β = size(V)
  z = size(A,2)
  @assert size(A) === (a,z,α) "Factor A has the wrong dimensions: need $((a,z,α)), got $(size(A))"
  @assert size(S) === (β,z,b) "Factor S has the wrong dimensions: need $((β,z,b)), got $(size(S))"

  if a*β*(α+b+1) < α*(b+1)*(a+β)
    # Order 1: AV[a,z,β] = A[a,z,α] * V[α,β], then W[a,b] = AV * S
    AV = mul!!(reshape(A, a*z, α), V, buffer=buffer[1])
    AV = reshape(AV, a, z, β)

    # Permute AV[a,z,β] → AV_permuted[a,β,z] to match S's row order (β,z).
    AV_permuted = permutedims!!(AV, (1,3,2), buffer=buffer[2])
    AV_permuted = reshape(AV_permuted, a, β*z)

    mul!(W, AV_permuted, reshape(S, β*z, b))
  else
    # Order 2: VS[α,z,b] = V[α,β] * S[β,z,b], then W[a,b] = A * VS
    VS = mul!!(V, reshape(S, β, z*b), buffer=buffer[1])
    VS = reshape(VS, α*z, b)

    # Permute A[a,z,α] → A_permuted[a,α,z] to match VS's row order (α,z).
    A_permuted = permutedims!!(A, (1,3,2), buffer=buffer[2])
    A_permuted = reshape(A_permuted, a, α*z)

    mul!(W, A_permuted, VS)
  end

  return W
end

function contract_sketch_core_backwards_buffers_size(z::Int, a::Int, α::Int, β::Int, b::Int)
  if a*β*(α+b+1) < α*(b+1)*(a+β)
    return (a*z*β, a*z*β)  # buffer[1] for AV, buffer[2] for AV_permuted
  else
    return (α*z*b, a*α*z)  # buffer[1] for VS, buffer[2] for A_permuted
  end
end

function contract_sketch_core_backwards!(W::AbstractArray{T,3}, A::AbstractArray{T,3}, B::AbstractArray{T,4}, S::AbstractArray{T,3}, V::AbstractArray{T,3};
                                buffer=(nothing,nothing,nothing)) where T

  # New layout: A = (a, z, α) vector core (L, I, R); B = (b, ζ, z, β) operator core (L, i_out, i_in, R).
  a,b,c = size(W)
  α,β,γ = size(V)
  ζ = size(B,2)
  z = size(B,3)
  @assert size(A) === (a,z,α)   "Factor A has the wrong dimensions: need $((a,z,α)), got $(size(A))"
  @assert size(B) === (b,ζ,z,β) "Factor B has the wrong dimensions: need $((b,ζ,z,β)), got $(size(B))"
  @assert size(S) === (γ,ζ,c)   "Factor S has the wrong dimensions: need $((γ,ζ,c)), got $(size(S))"

  if a*γ*(z*β*(α+ζ*b)+ζ*b*c) < α*c*(ζ*β*(γ+z*b) + z*a*b) # Order 1: ((V*A)*B)*S
  # Step 1: VA[a,z,β,γ] = A[a*z, α] * V[α,β,γ]
    VA = mul!!(reshape(A, a*z, α), reshape(V, α, β*γ), buffer=buffer[1])
    VA = reshape(VA, a, z, β, γ)

  # Permute dimensions: VA_permuted[a,γ,z,β] = VA[a,z,β,γ]
    VA_permuted = permutedims!!(VA, (1,4,2,3), buffer=buffer[2])
    VA_permuted = reshape(VA_permuted, a*γ,z*β)

  # Permute dimensions: B_permuted[z,β,ζ,b] = B[b,ζ,z,β]
    B_permuted = permutedims!!(B, (3,4,2,1), buffer=buffer[1])
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
    # Step 1: VS[α,β,ζ,c] = V[α,β,γ] * S[γ,ζ,c]
    VS = mul!!(reshape(V, α*β, γ), reshape(S, γ, ζ*c), buffer=buffer[1])
    VS = reshape(VS, α,β,ζ,c)

    # Permute dimensions: VS_permuted[ζ,β,α,c] = VS[α,β,ζ,c]
    VS_permuted = permutedims!!(VS, (3,2,1,4), buffer=buffer[2])
    VS_permuted = reshape(VS_permuted, ζ*β, α*c)

    # Permute dimensions: B_permuted[z,b,ζ,β] = B[b,ζ,z,β]
    B_permuted = permutedims!!(B, (3,1,2,4), buffer=buffer[1])
    B_permuted = reshape(B_permuted, z*b, ζ*β)

    # Step 2: VSB[z,b,α,c] = B_permuted[z,b,ζ,β] * VS_permuted[ζ,β,α,c]
    VSB = mul!!(B_permuted, VS_permuted, buffer=buffer[3])
    VSB = reshape(VSB, z,b,α,c)

    # A already (a, z, α) — reshape directly without permutedims.
    A_flat = reshape(A, a, z*α)

    # Permute dimensions: VSB_permuted[z,α,b,c] = VSB[z,b,α,c]
    VSB_permuted = permutedims!!(VSB, (1,3,2,4), buffer=buffer[2])
    VSB_permuted = reshape(VSB_permuted, z*α,b*c)

    # Step 3: W[a,b,c] = A_flat[a,z*α] * VSB_permuted[z*α,b*c]
    mul!(reshape(W, a,b*c), A_flat, VSB_permuted)
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

  # New layout: each A[i] is a backwards vector core (w[i]=L, z=I, v[i]=R).
  w = size(W)
  v = size(V)
  z = size(A[1],2)
  for i=1:N
    @assert size(A[i]) == (w[i],z,v[i])   "Factor A[$i] has the wrong dimensions: need $((w[i],z,v[i])), got $(size(A[i]))"
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
    # buffer[2]: A_permuted[αj,aj,z] = Aj (w,z,v) → (v,w,z)
    A_permuted = permutedims!!(A[j], (3,1,2), buffer=buffer[2])
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
      # buffer[2]: A_permuted[ai,αi,z] = Ai (w,z,v) → (w,v,z)
      A_permuted = permutedims!!(A[i], (1,3,2), buffer=buffer[2])
    end
    # buffer[3]: VA[ai,a/α≠ij,βj,z] = A_permuted[ai,αi,z] * VA_permuted[αi,α≠ij,βj,z]
    VA_intermediate = mul!!(A_permuted, reshape(VA_permuted,v[i],:,z), buffer=buffer[3])
    va[1] = w[i]
    if j == N+1
      # buffer[1]: B_permuted[βj,bj,z] = S[β,z,b]
      B_permuted = permutedims!!(S, (1,3,2), buffer=buffer[1])
    else
      # buffer[1]: B_permuted[βj,bj,z] = Aj (w,z,v) → (v,w,z)
      B_permuted = permutedims!!(A[j], (3,1,2), buffer=buffer[1])
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
      # buffer[2]: A_permuted[ai,αi,z] = Ai (w,z,v) → (w,v,z)
      A_permuted = permutedims!!(A[i], (1,3,2), buffer=buffer[2])
    end
    # buffer[3]: VA_intermediate[ai,a/α≠ij,βj,z] = Ai_permuted[ai,αi,z] * VA_permuted[αi,α≠ij,βj,z]
    VA_intermediate = mul!!(A_permuted, reshape(VA_permuted,v[i],:,z), buffer=buffer[3])
    va[1] = w[i]
    if j == N+1
      # buffer[1]: B_permuted[βj,z,bj] = S[β,z,b]
      B_permuted = S
    else
      # buffer[1]: B_permuted[βj,z,bj] = Aj (w,z,v) → (v,z,w)
      B_permuted = permutedims!!(A[j], (3,2,1), buffer=buffer[1])
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
      # buffer[2]: B_permuted[βj,z,bj] = Aj (w,z,v) → (v,z,w)
      B_permuted = permutedims!!(A[j], (3,2,1), buffer=buffer[2])
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
