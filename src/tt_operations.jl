using Base.Threads
using TensorOperations
import Base.+
import Base.-
import Base.*
import Base./
import LinearAlgebra.dot

################################################################################
# Annotated TT linear-algebra primitives (Julia)
################################################################################


"""
TTvector + constant
Add a scalar constant to every entry of the TT-vector, returning another TTvector.

Semantics / shape notes:
- We implement x + y by increasing TT ranks by 1 on interior bonds to accomodate
  the constant term without destroying the TT structure. This is a standard
  trick: represent the sum as a block-diagonal embedding in cores so that the
  original tensor and a constant tensor coexist in the enlarged TT-format.

- typejoin(T,S) determines the numeric type for the output cores when T and S
  differ (promotes types if needed).

- rks = x.ttv_rks .+ 1 : we temporarily increase all ranks by 1, then force
  boundary ranks to be 1 again (rks[1], rks[end] = 1,1). The added dimension
  is used to store the new "constant" component.

Performance notes:
- This is a memory-lean way to represent x + scalar: we do not densify the
  tensor; we only augment the TT cores.
"""
function +(x::TTvector{T,N},y::S) where {T<:Number,S<:Number,N}
    # Determine common numeric type for outputs
    R = typejoin(T,S)

    # New ranks: add 1 to every internal bond so we have room for the constant
    rks = x.ttv_rks .+ 1
    # Ensure boundary ranks remain 1 (TT boundary conditions)
    rks[1], rks[end] = 1, 1

    # zeros_tt is assumed to allocate an empty TTvector with given dims and ranks
    # out.ttv_vec[k] will be Array{R,3} of shape (n_k, rks[k], rks[k+1])
    out = zeros_tt(R, x.ttv_dims, rks)

    # Copy the original first core into the left-most slice of the new core.
    # Then set the new slice (where we store the constant contribution) to y.
    out.ttv_vec[1][:,:,1:x.ttv_rks[2]] = x.ttv_vec[1]           # copy existing block
    out.ttv_vec[1][:,:,rks[2]] .= y                             # set the "constant" slice

    # For interior cores, copy the old core into the upper-left block and put
    # identity-like entries (ones) at the new far corner so the constant term
    # propagates multiplicatively across the cores. This makes the added
    # constant equal to y * (1 * 1 * ... * 1) across cores.
    for k in 2:N-1
        # copy old core into the corresponding sub-block
        out.ttv_vec[k][:,1:x.ttv_rks[k],1:x.ttv_rks[k+1]] = x.ttv_vec[k]
        # set the corner entries to 1 so that constant propagates through
        out.ttv_vec[k][:,rks[k],rks[k+1]] .= 1
    end

    # Last core: copy the last core and set the final corner element(s) to 1
    out.ttv_vec[N][:,1:x.ttv_rks[N],1:x.ttv_rks[N+1]] = x.ttv_vec[N]
    out.ttv_vec[N][:,rks[N],rks[N+1]] .= 1

    return out
end

# Commutative scalar addition: scalar + TTvector
+(y::S, x::TTvector{T,N}) where {T<:Number,S<:Number,N} = x + y


"""
Addition of two TTvector
Elementwise sum of two TTvectors x and y (same dims).
We implement the standard direct-sum trick by stacking cores along the TT-ranks.
Resulting TT ranks are the sum of the ranks (r^out_k = r^x_k + r^y_k).
"""
function +(x::TTvector{T,N}, y::TTvector{T,N}) where {T<:Number,N}
    @assert x.ttv_dims == y.ttv_dims "Incompatible dimensions"

    d = x.N
    # Preallocate vector of cores (each core is a 3D array)
    ttv_vec = Array{Array{T,3},1}(undef, d)

    # New ranks: sum ranks of x and y at each bond. We'll enforce boundary ranks = 1
    rks = x.ttv_rks + y.ttv_rks
    rks[1] = 1
    rks[d+1] = 1

    # Initialize each core to the right shape in parallel to leverage threads.
    @threads for k in 1:d
        # core k shape: (n_k, rks[k], rks[k+1])
        ttv_vec[k] = zeros(T, x.ttv_dims[k], rks[k], rks[k+1])
    end

    @inbounds begin
        # Fill first core: put x's first core in the left block and y's in the
        # right block (concatenation along the rank dimension)
        ttv_vec[1][:, :, 1:x.ttv_rks[2]] = x.ttv_vec[1]
        ttv_vec[1][:, :, (x.ttv_rks[2]+1):rks[2]] = y.ttv_vec[1]

        # Fill interior cores: place x's core in the top-left block and y's in
        # the bottom-right block of the new core. Off-diagonal blocks remain 0
        # (this keeps the two components independent inside the TT).
        @threads for k in 2:(d-1)
            ttv_vec[k][:, 1:x.ttv_rks[k], 1:x.ttv_rks[k+1]] = x.ttv_vec[k]
            ttv_vec[k][:, (x.ttv_rks[k]+1):rks[k], (x.ttv_rks[k+1]+1):rks[k+1]] = y.ttv_vec[k]
        end

        # Last core: same concatenation idea (rank dimension collapses on the right)
        ttv_vec[d][:, 1:x.ttv_rks[d], 1] = x.ttv_vec[d]
        ttv_vec[d][:, (x.ttv_rks[d]+1):rks[d], 1] = y.ttv_vec[d]
    end

    # Construct TTvector: note we keep x.ttv_dims and pass ttv_vec and new ranks
    return TTvector{T,N}(d, ttv_vec, x.ttv_dims, rks, zeros(Int64, d))
end


"""
Addition of two TToperators
Analogous to TTvector addition but for TToperators where each core is 4D:
shape [out_dim, in_dim, left_rank, right_rank].
We combine ranks in the same direct-sum manner.
"""
function +(x::TToperator{T,N}, y::TToperator{T,N}) where {T<:Number,N}
    @assert x.tto_dims == y.tto_dims "Incompatible dimensions"

    d = x.N
    tto_vec = Array{Array{T,4},1}(undef, d)
    rks = x.tto_rks + y.tto_rks
    rks[1] = 1
    rks[d+1] = 1

    # Initialize cores in parallel
    @threads for k in 1:d
        # each operator core: (n_k_out, n_k_in, r_k, r_{k+1})
        tto_vec[k] = zeros(T, x.tto_dims[k], x.tto_dims[k], rks[k], rks[k+1])
    end

    @inbounds begin
        # first core: copy x and y into disjoint rank blocks
        tto_vec[1][:,:,:, 1:x.tto_rks[1+1]] = x.tto_vec[1]
        tto_vec[1][:,:,:, (x.tto_rks[2]+1):rks[2]] = y.tto_vec[1]

        # interior cores: top-left block <- x, bottom-right block <- y
        @threads for k in 2:(d-1)
            tto_vec[k][:,:, 1:x.tto_rks[k], 1:x.tto_rks[k+1]] = x.tto_vec[k]
            tto_vec[k][:,:, (x.tto_rks[k]+1):rks[k], (x.tto_rks[k+1]+1):rks[k+1]] = y.tto_vec[k]
        end

        # last core
        tto_vec[d][:,:, 1:x.tto_rks[d], 1] = x.tto_vec[d]
        tto_vec[d][:,:, (x.tto_rks[d]+1):rks[d], 1] = y.tto_vec[d]
    end

    return TToperator{T,N}(d, tto_vec, x.tto_dims, rks, zeros(Int64, d))
end


# ------------------------------------------------------------------------
# Matrix-vector multiplication in TT format: y = A * v
# - A is a TToperator, with cores A.tto_vec[k][i_k, j_k, α_{k-1}, α_k]
# - v is a TTvector, with cores v.ttv_vec[k][j_k, ν_{k-1}, ν_k]
# The resulting TTvector y has TT-ranks equal to the Kronecker product of
# the operator ranks and vector ranks: y.ranks[k] = A.ranks[k] * v.ranks[k].
#
# Implementation details:
# - We allocate y = zeros_tt(T, dims, A.tto_rks .* v.ttv_rks) (elementwise product of ranks)
# - For each core k we build a temporary view yvec_temp with shape:
#     (n_k, A_r_k, v_r_k, A_r_{k+1}, v_r_{k+1})
#   which is a reshaping of the stored 3D core into a 5D view to match indices.
# - Then we use a tensor contraction (@tensoropt macro) to compute
#     yvec_temp[i_k, α_{k-1}, ν_{k-1}, α_k, ν_k] =
#         sum_{j_k} A.tto_vec[k][i_k, j_k, α_{k-1}, α_k] * v.ttv_vec[k][j_k, ν_{k-1}, ν_k]
#
# - This is the canonical local contraction for operator*vector in the TT format.
# - @inbounds and @simd / @threads are used to help performance (careful: ensure
#   that @tensoropt macro is thread-safe in your environment).
function *(A::TToperator{T,N}, v::TTvector{T,N}) where {T<:Number,N}
    @assert A.tto_dims == v.ttv_dims "Incompatible dimensions"

    # allocate output TTvector where ranks are elementwise product:
    y = zeros_tt(T, A.tto_dims, A.tto_rks .* v.ttv_rks)

    @inbounds begin @simd for k in 1:v.N
        # Create a reshaped view of the k-th core of the result so we can write into
        # it with indices matching the contraction pattern. The storage layout of
        # y.ttv_vec[k] is (n_k, r_left, r_right) where r_left = A_r_k * v_r_k, etc.
        yvec_temp = reshape(y.ttv_vec[k], (y.ttv_dims[k], A.tto_rks[k], v.ttv_rks[k], A.tto_rks[k+1], v.ttv_rks[k+1]))

        # The @tensoropt macro (assumed provided by TensorOperations or a custom macro)
        # performs the contraction in an efficient manner:
        @tensoropt((νₖ₋₁,νₖ), yvec_temp[iₖ,αₖ₋₁,νₖ₋₁,αₖ,νₖ] = A.tto_vec[k][iₖ,jₖ,αₖ₋₁,αₖ]*v.ttv_vec[k][jₖ,νₖ₋₁,νₖ])
        #
        # index notation:
        # - iₖ : physical output index for operator core (1..n_k)
        # - jₖ : physical input index matching v core (1..n_k)
        # - αₖ₋₁, αₖ : operator TT-ranks for left/right bonds
        # - νₖ₋₁, νₖ : vector TT-ranks for left/right bonds
    end end

    return y
end

# ------------------------------------------------------------------------
# Matrix-matrix multiplication in TT format: (A * B) where A, B are TToperators
# - Each result core Y[k] is constructed such that its ranks are the product of
#   the corresponding ranks of A and B: new_rks[k] = A_rks[k] * B_rks[k].
#
# Implementation outline:
# 1) Allocate a temporary Y[k] with shape (n_k, n_k, A_rk*B_rk, A_rk+1*B_rk+1)
# 2) Compute the local contraction:
#      M_temp[i_k, j_k, α_{k-1}, β_{k-1}, α_k, β_k] =
#          sum_{z} A.tto_vec[k][i_k, z, α_{k-1}, α_k] * B.tto_vec[k][z, j_k, β_{k-1}, β_k]
#    where z indexes the shared inner physical dimension of the operator cores
#    (the in/out physical index of the operator).
# 3) Reshape / store results into Y[k].
#
# This nested contraction is implemented with triple loops and a @tensor macro.
function *(A::TToperator{T,N}, B::TToperator{T,N}) where {T<:Number,N}
    @assert A.tto_dims == B.tto_dims "Incompatible dimensions"

    d = A.N
    A_rks = A.tto_rks
    B_rks = B.tto_rks

    # Preallocate result cores in a vector comprehension; each Y[k] is a 4D array
    Y = [zeros(T, A.tto_dims[k], A.tto_dims[k], A_rks[k]*B_rks[k], A_rks[k+1]*B_rks[k+1]) for k in eachindex(A.tto_dims)]

    @inbounds @simd for k in eachindex(Y)
        # Reshape the target core Y[k] into a 6D temporary to match contraction indices:
        M_temp = reshape(Y[k], A.tto_dims[k], A.tto_dims[k], A_rks[k], B_rks[k], A_rks[k+1], B_rks[k+1])

        # We iterate over physical indices (i_k, j_k) and perform the contraction
        # over the inner physical index z. Note: we keep @simd on loops where
        # it is beneficial; the heavy lifting is done by @tensor contraction.
        @simd for jₖ in size(M_temp, 2)
            @simd for iₖ in size(M_temp, 1)
                @tensor M_temp[iₖ, jₖ, αₖ₋₁, βₖ₋₁, αₖ, βₖ] = A.tto_vec[k][iₖ, z, αₖ₋₁, αₖ] * B.tto_vec[k][z, jₖ, βₖ₋₁, βₖ]
            end
        end
    end

    # Return a new TToperator with ranks multiplied bondwise
    return TToperator{T,N}(d, Y, A.tto_dims, A.tto_rks .* B.tto_rks, zeros(Int64, d))
end

# variadic multiplication: A * B * C ... associate to the left by repeated calls
*(A::TToperator{T,N}, B...) where {T,N} = *(A, *(B...))


"""
Linear combination (weighted sum) of TTvectors stored in an array A:
Given A = [A1, A2, ...] (an Array{TTvector,1}) and a coefficient vector x,
compute sum_i x[i] * A[i].

This treats the array A as a basis of TTvectors and forms the linear combination
using scalar multiplications and additions. It assumes operations + and * are
defined for TTvectors and scalars (they are in this file).
"""
function *(A::Array{TTvector{T,N},1}, x::Vector{T}) where {T,N}
    out = x[1] * A[1]            # scale first basis vector
    for i in 2:length(A)
        out = out + x[i] * A[i] # accumulate remaining scaled basis vectors
    end
    return out
end

"""
Pointwise (Hadamard) product of two TTvectors.
We return a TTvector whose cores are obtained by taking the Kronecker product
of corresponding slices of the input cores. This corresponds to the
componentwise product of the represented full tensors.

- Output ranks at bond k are x.ttv_rks[k] * y.ttv_rks[k]
- For each physical index i_k we take kron(x_core[i_k,:,:], y_core[i_k,:,:]) to
  get the combined core slice.

Note: this operation can cause rank explosion; consider rounding/truncation
afterwards if ranks become too large.
"""
function *(x::TTvector{T,N}, y::TTvector{T,N}) where {T<:Number,N}
    # create output with ranks equal to elementwise product of input ranks
    out = zeros_tt(T, x.ttv_dims, x.ttv_rks .* y.ttv_rks)

    for k in 1:N
        # iterate over physical indices for core k and build Kronecker slices
        for iₖ in 1:x.ttv_dims[k]
            # kron of the two (left_rank × right_rank) matrices yields the
            # combined (left_rank * left_rank) × (right_rank * right_rank) matrix
            out.ttv_vec[k][iₖ, :, :] = kron(x.ttv_vec[k][iₖ, :, :], y.ttv_vec[k][iₖ, :, :])
        end
    end

    return out
end

# ------------------------------------------------------------------------
# dot returns the scalar dot product of two TTvectors A and B.
#
# Algorithmic idea (standard TT inner product algorithm):
# - We maintain a small matrix `out` of size max(A_rk) × max(B_rk) that holds
#   the accumulated contraction of processed cores.
# - Initialize out[1,1] = 1 (scalar) and iteratively fold cores k=1..N:
#       M = contraction involving A_core[k], B_core[k] and previous out block.
# - At the end out[1,1] contains the scalar inner product.
#
# The contraction inside the loop uses indices:
#    M[a,b] = sum_z,α,β A_core[k][z,α,a] * (B_core[k][z,β,b] * out[α,β])
# which corresponds to contracting physical index z and left ranks α,β.
function dot(A::TTvector{T,N}, B::TTvector{T,N}) where {T<:Number,N}
    @assert A.ttv_dims == B.ttv_dims "TT dimensions are not compatible"

    A_rks = A.ttv_rks
    B_rks = B.ttv_rks

    # `out` stores accumulated contraction (max ranks ensure enough space)
    out = zeros(T, maximum(A_rks), maximum(B_rks))
    out[1,1] = one(T)  # start with scalar 1

    @inbounds for k in eachindex(A.ttv_dims)
        # view into current block shaped (R^A_{k}, R^B_{k})
        M = @view(out[1:A_rks[k+1], 1:B_rks[k+1]])

        # contraction over physical index and previous block `out`
        @tensor M[a,b] = A.ttv_vec[k][z, α, a] * (B.ttv_vec[k][z, β, b] * out[1:A_rks[k], 1:B_rks[k]][α, β])
        # after contracting this core, M becomes the new "out" for next iteration
    end

    # return the scalar in out[1,1]
    return out[1,1]::T
end


"""
dot_par(A, B) : parallelized dot.
Compute the dot product using a two-stage algorithm that builds and multiplies
local contribution matrices in parallel and reduces them sequentially.

Algorithm:
- For each core k build a 4D local tensor M (left_rank_prev, right_rank_prev, left_rank_next, right_rank_next)
  storing the corewise contraction for that site.
- Reshape each M into a matrix Y[k] of shape (R^A_k * R^B_k, R^A_{k+1} * R^B_{k+1}).
  This is the local transfer matrix for that core.
- Multiply Y[d] (last) then successively multiply by Y[d-1], ..., Y[1] to reduce to a scalar.
- The per-core construction is parallelized using @threads because each core's
  local contraction is independent.
"""
function dot_par(A::TTvector{T,N}, B::TTvector{T,N}) where {T<:Number,N}
    @assert A.ttv_dims == B.ttv_dims "TT dimensions are not compatible"
    d = length(A.ttv_dims)

    # Preallocate array Y of matrices (one per core)
    Y = Array{Array{T,2},1}(undef, d)
    A_rks = A.ttv_rks
    B_rks = B.ttv_rks

    # C will hold the running vector that we propagate backward (a flattened block)
    C = zeros(T, maximum(A_rks .* B_rks))

    # Build each local transfer matrix in parallel
    @threads for k in 1:d
        # M has shape (A_rk, B_rk, A_rk+1, B_rk+1)
        M = zeros(T, A_rks[k], B_rks[k], A_rks[k+1], B_rks[k+1])

        # Contract the physical index z to build M for this site
        @tensor M[a, b, c, d] = A.ttv_vec[k][z, a, c] * B.ttv_vec[k][z, b, d]

        # Reshape into matrix form: rows = (A_rk * B_rk), cols = (A_rk+1 * B_rk+1)
        Y[k] = reshape(M, A_rks[k] * B_rks[k], A_rks[k+1] * B_rks[k+1])
    end

    # Initialize running vector C with the last transfer matrix flattened
    @inbounds C[1:length(Y[d])] = Y[d][:]

    # Multiply the transfer matrices backwards to reduce to scalar: C = Y[1] * (Y[2] * (...))
    for k in d-1:-1:1
        @inbounds C[1:size(Y[k], 1)] = Y[k] * C[1:size(Y[k], 2)]
    end

    return C[1]::T
end


# Scalar * TTvector: multiply one core (the first non-orthogonal?) by scalar
# The code finds an index i where A.ttv_ot equals 0 and multiplies only that core by a.
# This is a somewhat unconventional optimization: it avoids scaling all cores if
# there is metadata that marks a "root" core for scalar embedding.
function *(a::S, A::TTvector{R,N}) where {S<:Number, R<:Number, N}
    T = typejoin(typeof(a), R)

    # If scalar is zero return zero TTvector with minimal ranks (all ones)
    if iszero(a)
        return zeros_tt(T, A.ttv_dims, ones(Int64, A.N + 1))
    else
        # find index of first zero in orthogonality mask (ttv_ot) - the core to scale
        i = findfirst(isequal(0), A.ttv_ot)
        X = copy(A.ttv_vec)     # shallow copy of core array (cores themselves are arrays)
        X[i] = a * X[i]         # scale only the chosen core
        return TTvector{T,N}(A.N, X, A.ttv_dims, A.ttv_rks, A.ttv_ot)
    end
end

# Scalar * TToperator: analogous to above but for operators
function *(a::S, A::TToperator{R,N}) where {S<:Number, R<:Number, N}
    i = findfirst(isequal(0), A.tto_ot)
    T = typejoin(typeof(a), R)
    X = copy(A.tto_vec)
    X[i] = a * X[i]
    return TToperator{T,N}(A.N, X, A.tto_dims, A.tto_rks, A.tto_ot)
end

# Vector subtraction implemented via scalar multiplication and addition
function -(A::TTvector{T,N}, B::TTvector{T,N}) where {T<:Number,N}
    return *(-1.0, B) + A
end

function -(A::TToperator{T,N}, B::TToperator{T,N}) where {T<:Number,N}
    return *(-1.0, B) + A
end

# Division of TTvector by scalar
function /(A::TTvector, a)
    return 1 / a * A
end


"""
outer_product(x, y) : builds a TToperator representing x * y' (outer product).
Given TTvectors x and y, construct a TToperator whose cores are obtained by
contracting x and conj(y) at each site and arranging ranks to be the product
of the individual ranks. This produces an operator whose action on a vector z
is (x * (y' * z)) i.e. projection onto y followed by scaling by x.
"""
function outer_product(x::TTvector{T,N}, y::TTvector{T,N}) where {T<:Number,N}
    # allocate per-site operator-cores Y[k] with shapes:
    # (n_k, n_k, x_rk * y_rk, x_rk+1 * y_rk+1)
    Y = [zeros(T, x.ttv_dims[k], x.ttv_dims[k], x.ttv_rks[k] * y.ttv_rks[k], x.ttv_rks[k+1] * y.ttv_rks[k+1]) for k in eachindex(x.ttv_dims)]

    @inbounds @simd for k in eachindex(Y)
        # reshape Y[k] into a 6D temporary M_temp so we can express the
        # contraction in index notation conveniently.
        M_temp = reshape(Y[k], x.ttv_dims[k], x.ttv_dims[k], x.ttv_rks[k], y.ttv_rks[k], x.ttv_rks[k+1], y.ttv_rks[k+1])

        # For every pair of physical indices (i_k, j_k) we set the block
        # corresponding to the pair of left- and right-ranks to the outer
        # product of x_core and conj(y_core). The conj ensures correct
        # adjoint-like behavior for the outer product operator.
        @simd for jₖ in size(M_temp, 2)
            @simd for iₖ in size(M_temp, 1)
                @tensor M_temp[iₖ, jₖ, αₖ₋₁, βₖ₋₁, αₖ, βₖ] = x.ttv_vec[k][iₖ, αₖ₋₁, αₖ] * conj(y.ttv_vec[k][jₖ, βₖ₋₁, βₖ])
            end
        end
    end

    # Return TToperator with multiplied ranks (elementwise product of vector ranks)
    return TToperator{T,N}(x.N, Y, x.ttv_dims, x.ttv_rks .* y.ttv_rks, zeros(Int64, x.N))
end
