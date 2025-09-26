using TensorTrains
using TensorOperations
using Random
import LinearAlgebra.norm
import Base: show


function show(io::IO, X::TTvector)
    print(io, "TTvector of size ", X.ttv_dims, " and ranks ", X.ttv_rks)
end


function full(X::TTvector)
    Y = X.ttv_vec[1]
    Y = reshape(Y,(size(Y,1),size(Y,3)))
    for y in (X.ttv_vec)[2:end]
        R = reshape(y,(size(y,2), size(y,1)*size(y,3))) # ri-1 x ni ri
        Y = Y*R
        Y = reshape(Y,(size(Y,1)*size(Y,2)÷size(y,3), size(y,3))) # (n1...ni) x ri
    end
    Y = reshape(Y,(X.ttv_dims...))
    return Y
end


function vecnorm(A::Array{T,N}, p::Int, dim::Int=1) where {T<:Number,N}
    return (sum(abs.(A).^p, dims=dim)).^(1/p)
end


# - If you need different behaviour (e.g. keep right boundary open) change the
#   initialization of C accordingly (e.g. use `Matrix{T}(I, rA[end], rB[end])`).
function tt_dot(A::TTvector{T1,N}, B::TTvector{T2,N}) where {T1<:Number,T2<:Number,N}
    @assert A.ttv_dims == B.ttv_dims "TT dimensions are not compatible"

    if T1 == ComplexF64 || T2 == ComplexF64
        T = ComplexF64
    else
        T = Float64
    end
    A_rks = A.ttv_rks
    B_rks = B.ttv_rks

    # Start by summing over the rightmost boundary ranks (and any trailing cores).
    # Using ones(T, rA_end, rB_end) effectively contracts those rightmost ranks
    # (i.e., performs an unweighted sum across them). This makes the final C
    # depend only on the left boundary ranks.
    C = ones(T, A_rks[end], B_rks[end])

    

    # Backward recursion: process cores from N down to 1
    @inbounds for k in N:-1:1
        # allocate next contracted matrix of size (rA[k], rB[k])
        Cnew = zeros(T, A_rks[k], B_rks[k])

        # perform the local contraction for site k:
        #   Cnew[α, β] = Σ_{z, αp, βp} A_k[z, α, αp] * B_k[z, β, βp] * C[αp, βp]
        @tensor Cnew[a, b] = A.ttv_vec[k][z, a, ap] * (B.ttv_vec[k][z, b, bp] * C[ap, bp])

        # advance the recursion
        C = Cnew
    end

    # If the result is 1×1 return a scalar, otherwise return the matrix
    if size(C, 1) == 1 && size(C, 2) == 1
        return C[1, 1]::T
    else
        return C
    end
end

function tt_randn(rng::AbstractRNG,dims::NTuple{N,Int64},rks::Vector{Int64};orthogonal=false,right=true,T::Type{<:Number}=Float64) where {N}

    #= println(rks)
    println(r_and_d_to_rks(rks,dims;rmax = maximum(rks))) =#

	y = zeros_tt(T,dims,rks)
	@simd for i in eachindex(y.ttv_vec)
		y.ttv_vec[i] = randn(rng, T,dims[i], rks[i], rks[i+1])
		if orthogonal
			if right
                q,_ = qr(reshape(permutedims(y.ttv_vec[i],(1,3,2)),dims[i]*rks[i+1],rks[i]))
                y.ttv_vec[i] = permutedims(reshape(Matrix(q),dims[i],rks[i+1],rks[i]),(1,3,2))
			else
                q,_ = qr(reshape(y.ttv_vec[i],dims[i]*rks[i],rks[i+1]))
                y.ttv_vec[i] = reshape(Matrix(q),dims[i],rks[i],rks[i+1])
			end
            y.ttv_vec[i] *= sqrt(dims[i]*rks[i+1]/rks[i])
        else
            y.ttv_vec[i] /= sqrt(rks[i]) 
		end
	end
	return y
end