"""
    contract_sketch_core_forwards!(W, A, S, V; buffer=(nothing,nothing), timer=TimerOutput())
    contract_sketch_core_forwards!(W, A, B, S, V; buffer=(nothing,nothing,nothing), timer=TimerOutput())

Efficient contraction kernel for forward tensor train sketching:
W[a,b] = A[z,α,a] * S[β,z,b] * V[α,β]
or
W[a,b,c] = A[ζ,α,a] * B[z,ζ,β,b] * S[γ,z,c] * V[α,β,γ]

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
  `B::AbstractArray{T,4}`: Operator tensor [z,ζ,β,b] (forward TTO format)
- `S::AbstractArray{T,3}`: Sketch block tensor [γ,z,c] (sketch block format)
- `V::AbstractArray{T,3}`: Connection array [α,β,γ] (previous sketch weights)

# Optional arguments
- `buffer`: Optional preallocated buffer arrays
- `timer::TimerOutput`: Optional timer for performance profiling

"""
function contract_sketch_core_forwards!(W::AbstractMatrix{T}, A::AbstractArray{T,3}, S::AbstractArray{T,3}, V::AbstractMatrix{T};
                                buffer=(nothing,nothing)) where T

  # A has the new TT layout (L, I, R) = (α, z, a).
  a,b = size(W)
  α,β = size(V)
  z = size(A,2)
  @assert size(A) === (α,z,a) "Factor A has the wrong dimensions: need $((α,z,a)), got $(size(A))"
  @assert size(S) === (β,z,b) "Factor S has the wrong dimensions: need $((β,z,b)), got $(size(S))"

  if a*(β+1)*(α+b) < α*b*(a+β+1)
    # Order 1: ((V'×A)×S) — A already in (α, z, a), no permute needed.
    A_flat = reshape(A, α, z*a)
    VA = mul!!(transpose(V), A_flat, buffer=buffer[2])
    VA = reshape(VA, β*z, a)
    mul!(W, transpose(VA), reshape(S, β*z, b))
  else
    # Order 2: ((V×S)×A) — VS reshape (α*z, b) matches A reshape (α*z, a) row-wise.
    VS = mul!!(V, reshape(S, β, z*b), buffer=buffer[1])
    mul!(W, transpose(reshape(A, α*z, a)), reshape(VS, α*z, b))
  end

  return W
end

function contract_sketch_core_forwards_buffers_size(z::Int, α::Int, a::Int, β::Int, b::Int)
  if a*(β+1)*(α+b) < α*b*(a+β+1)
    return (0, β*z*a)  # buffer[2] for VA = V'×A; buffer[1] unused.
  else
    return (α*z*b, 0)  # buffer[1] for VS = V×S; buffer[2] unused.
  end
end

"""
    contract_sketch_core_forwards!(W, A, S, V; buffer=(nothing,nothing))

Batched (p-axis) version of the forward A·S·V kernel.
  W: (a, b, p) — output, modified in-place
  A: (α, z, a) — shared across p
  S: (β, z, b, p) — block sketches, per-p
  V: (α, β, p) — previous sketch weights, per-p

Symmetric to the backward batched kernel: contracts α first via one batched
`V_batched' × A_flat` gemm of shape `(β*p, α) × (α, z*a)`, then runs a per-p
inner gemm. See backward variant for ordering rationale.
"""
function contract_sketch_core_forwards!(W::AbstractArray{T,3}, A::AbstractArray{T,3}, S::AbstractArray{T,4}, V::AbstractArray{T,3};
                                buffer=(nothing,nothing)) where T
  a, b, p = size(W)
  α, β, _ = size(V)
  z = size(A, 2)
  @assert size(A) === (α, z, a)        "Factor A has the wrong dimensions: need $((α,z,a)), got $(size(A))"
  @assert size(V) === (α, β, p)        "Factor V has the wrong dimensions: need $((α,β,p)), got $(size(V))"
  @assert size(S) === (β, z, b, p)     "Factor S has the wrong dimensions: need $((β,z,b,p)), got $(size(S))"

  # Step 1: VA[β*p, z*a] = V_batched'[β*p, α] × A_flat[α, z*a] — one batched gemm
  # (matches the unbatched order-1 path: tall left factor with α reduced first).
  V_mat   = reshape(V, α, β*p)
  A_flat  = reshape(A, α, z*a)
  VA      = mul!!(transpose(V_mat), A_flat, buffer=buffer[2])
  VA_4d   = reshape(VA, β, p, z, a)
  # Permute (β, p, z, a) → (β, z, a, p) so per-p slice is contiguous (β*z, a).
  VA_perm = permutedims!!(VA_4d, (1, 3, 4, 2), buffer=buffer[1])
  VA_perm_3d = reshape(VA_perm, β*z, a, p)

  # Step 2 per-p: W[:, :, j] = VA_perm[:, :, j]' × reshape(S[:, :, :, j], β*z, b)
  @inbounds for j in 1:p
    @views mul!(W[:, :, j], transpose(VA_perm_3d[:, :, j]), reshape(S[:, :, :, j], β*z, b))
  end
  return W
end

function contract_sketch_core_forwards_batched_buffers_size(z::Int, α::Int, a::Int, β::Int, b::Int, p::Int)
  return (β*z*a*p, β*z*a*p)  # buffer[1] for VA_perm, buffer[2] for VA
end

function contract_sketch_core_forwards!(W::AbstractArray{T,3}, A::AbstractArray{T,3}, B::AbstractArray{T,4}, S::AbstractArray{T,3}, V::AbstractArray{T,3};
                                buffer=(nothing,nothing,nothing)) where T

  # New layout: A = (α, ζ, a) vector core (L, I, R); B = (β, z, ζ, b) operator core (L, i_out, i_in, R).
  a,b,c = size(W)
  α,β,γ = size(V)
  z = size(B,2)
  ζ = size(B,3)
  @assert size(A) === (α,ζ,a)   "Factor A has the wrong dimensions: need $((α,ζ,a)), got $(size(A))"
  @assert size(B) === (β,z,ζ,b) "Factor B has the wrong dimensions: need $((β,z,ζ,b)), got $(size(B))"
  @assert size(S) === (γ,z,c)   "Factor S has the wrong dimensions: need $((γ,z,c)), got $(size(S))"

  if ζ*a*α*β*γ + a*γ*ζ*β*z*b + a*b*γ*z*c < α*β*γ*z*c + ζ*b*z*β*α*c + a*ζ*α*b*c # Order 1: ((V*A)*B)*S
    # Permute dimensions: A_permuted[ζ,a,α] = A[α,ζ,a]
    A_permuted = permutedims!!(A, (2,3,1), buffer=buffer[1])
    # Step 1: VA[ζ,a,β,γ] = A_permuted[ζ*a, α] * V[α,β,γ]
    VA = mul!!(reshape(A_permuted, ζ*a, α), reshape(V, α, β*γ), buffer=buffer[2])
    VA = reshape(VA, ζ, a, β, γ)

    # Permute dimensions: VA_permuted[a,γ,ζ,β] = VA[ζ,a,β,γ]
    VA_permuted = permutedims!!(VA, (2,4,1,3), buffer=buffer[1])
    VA_permuted = reshape(VA_permuted, a*γ,ζ*β)

    # Permute dimensions: B_permuted[ζ,β,z,b] = B[β,z,ζ,b]
    B_permuted = permutedims!!(B, (3,1,2,4), buffer=buffer[2])
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
    # Step 1: VS[α,β,z,c] = V[α,β,γ] * S[γ,z,c]
    VS = mul!!(reshape(V, α*β, γ), reshape(S, γ, z*c), buffer=buffer[1])
    VS = reshape(VS, α,β,z,c)

    # Permute dimensions: VS_permuted[z,β,α,c] = VS[α,β,z,c]
    VS_permuted = permutedims!!(VS, (3,2,1,4), buffer=buffer[2])
    VS_permuted = reshape(VS_permuted, z*β, α*c)

    # Permute dimensions: B_permuted[ζ,b,z,β] = B[β,z,ζ,b]
    B_permuted = permutedims!!(B, (3,4,2,1), buffer=buffer[1])
    B_permuted = reshape(B_permuted, ζ*b, z*β)

    # Step 2: VBS[ζ,b,α,c] = B_permuted[ζ,b,z,β] * VS_permuted[z,β,α,c]
    VBS = mul!!(B_permuted, VS_permuted, buffer=buffer[3])
    VBS = reshape(VBS, ζ,b,α,c)

    # Permute dimensions: A_permuted[a,ζ,α] = A[α,ζ,a]
    A_permuted = permutedims!!(A, (3,2,1), buffer=buffer[1])
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

  # New layout: each A[i] is a vector core (L=v[i], I=z, R=w[i]).
  w = size(W)
  v = size(V)
  z = size(A[1],2)
  for i=1:N
    @assert size(A[i]) == (v[i],z,w[i])   "Factor A[$i] has the wrong dimensions: need $((v[i],z,w[i])), got $(size(A[i]))"
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
    # buffer[2]: A_permuted[αj,aj,z] = Aj (v,z,w) → (v,w,z)
    A_permuted = permutedims!!(A[j], (1,3,2), buffer=buffer[2])
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
      # buffer[2]: A_permuted[ai,αi,z] = Ai (v,z,w) → (w,v,z)
      A_permuted = permutedims!!(A[i], (3,1,2), buffer=buffer[2])
    end
    # buffer[3]: VA[ai,a/α≠ij,βj,z] = A_permuted[ai,αi,z] * VA_permuted[αi,α≠ij,βj,z]
    VA_intermediate = mul!!(A_permuted, reshape(VA_permuted,v[i],:,z), buffer=buffer[3])
    va[1] = w[i]
    if j == N+1
      # buffer[1]: B_permuted[βj,bj,z] = S[β,z,b]
      B_permuted = permutedims!!(S, (1,3,2), buffer=buffer[1])
    else
      # buffer[1]: B_permuted[βj,bj,z] = Aj (v,z,w) → (v,w,z)
      B_permuted = permutedims!!(A[j], (1,3,2), buffer=buffer[1])
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
      # buffer[2]: A_permuted[ai,αi,z] = Ai (v,z,w) → (w,v,z)
      A_permuted = permutedims!!(A[i], (3,1,2), buffer=buffer[2])
    end
    # buffer[3]: VA_intermediate[ai,a/α≠ij,βj,z] = Ai_permuted[ai,αi,z] * VA_permuted[αi,α≠ij,βj,z]
    VA_intermediate = mul!!(A_permuted, reshape(VA_permuted,v[i],:,z), buffer=buffer[3])
    va[1] = w[i]
    if j == N+1
      # buffer[1]: B_permuted[βj,z,bj] = S[β,z,b]
      B_permuted = S
    else
      # buffer[1]: B_permuted[βj,z,bj] = Aj (v,z,w) — already in (v,z,w) order; no permute needed.
      B_permuted = A[j]
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
      # buffer[2]: Bj_permuted[βj,z,bj] = Aj (v,z,w) — already in (v,z,w) order.
      B_permuted = A[j]
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
