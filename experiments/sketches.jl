using TensorTrains
using LinearAlgebra
using StaticArrays


function TTPR(rng::AbstractRNG, X::SizedArray{Tuple{M},TTvector{T0,N}}, dims::NTuple{N,Int64}, rk::Vector{Int64}, P::Int64, R::Int64; normalization::AbstractString="none", batch::Int64=M, orthogonal=false, right=true, T::Type{<:Number}=Float64) where {T0,N,M}
    #k = Int(R*P)
    k = Int(rk[1]*P)
    Ω = Vector{TTvector{T,N}}(undef, k)
    ttr = zeros(T, k, batch)

    #rks = length(rk)==1 ? vcat(P, fill(rk[1], N-1), 1) : rk
    rks = length(rk)==1 ? vcat(fill(rk[1], N), 1) : rk

    if orthogonal
        tmp = [reverse(cumprod(reverse(dims)))..., 1]
        rks = [min(rks[i], tmp[i]) for i in eachindex(tmp)]
    end

    for i in 1:P
        cores = tt_randn(rng, dims, rks, normalization=normalization, orthogonal=orthogonal, right=right, T=T).ttv_vec
        for j = 1:N
            if orthogonal
                #TODO FIX?
                #cores[j] *= sqrt(dims[j])/((rks[j]*rks[j+1])^(0.25))
                #cores[j] *= sqrt(rks[j+1])/((rks[j]*rks[j+1])^(0.25)) #Orthogonal(?)
                cores[j] *= 1/((rks[j]*rks[j+1])^(0.25)) #Pure Gaussian (same as non-orthogonal)
            else
                cores[j] *= 1/((rks[j]*rks[j+1])^(0.25))
            end
        end
        ω = TTvector{T,N}(N, cores, dims, rks, zeros(Int64, N))
        Ω[i] = ω

        for j = 1:batch
            if rk[1] == 1 # Julia doesn't like assignments like ttr[:,1] = scalar
                ttr[i,j] = tt_dot(X[j], ω)/sqrt(k)
            else
                ttr[(i-1)*rk[1]+1:i*rk[1], j] = tt_dot(X[j], ω)/sqrt(k)
            end
        end       
    end
    return ttr, Ω
end



function TTR(rng::AbstractRNG, X::SizedArray{Tuple{M},TTvector{T0,N}}, dims::NTuple{N,Int64}, rk::Vector{Int64}, k::Int64; normalization::AbstractString="none", batch::Int64=M, orthogonal=false, right=true, T::Type{<:Number}=Float64) where {T0,N,M}
    Ω = Vector{TTvector{T,N}}(undef, k)
    ttr = zeros(T, k, batch)

    rks = length(rk)==1 ? vcat(1, fill(rk[1], N-1), 1) : rk

    if orthogonal
        tmp = [reverse(cumprod(reverse(dims)))..., 1]
        rks = [min(rks[i], tmp[i]) for i in eachindex(tmp)]
    end

    for i in 1:k
        cores = tt_randn(rng, dims, rks, normalization=normalization, orthogonal=orthogonal, right=right, T=T).ttv_vec
        for j = 1:N
            if orthogonal
                #TODO FIX?
                #cores[j] *= sqrt(dims[j])/((rks[j]*rks[j+1])^(0.25))
                #cores[j] *= sqrt(rks[j+1])/((rks[j]*rks[j+1])^(0.25)) #Orthogonal(?)
                cores[j] *= 1/((rks[j]*rks[j+1])^(0.25)) #Pure Gaussian (same as non-orthogonal)
            else
                cores[j] *= 1/((rks[j]*rks[j+1])^(0.25))
            end
        end
        ω = TTvector{T,N}(N, cores, dims, rks, zeros(Int64, N))
        Ω[i] = ω

        for j = 1:batch
            ttr[i,j] = tt_dot(X[j], ω)/sqrt(k)
        end       
    end
    return ttr, Ω
end


function GTT(rng::AbstractRNG, X::SizedArray{Tuple{M},TTvector{T0,N}}, dims::NTuple{N,Int64}, k::Int64; normalization::AbstractString="none", batch::Int64=M, orthogonal::Bool=false, right::Bool=true, T::Type{<:Number}=Float64) where {T0,N,M}
    
    gtt = zeros(T, k, batch)

    if orthogonal
        rks = [min(k, i) for i in [reverse(cumprod(reverse(dims)))..., 1]]
    else
        rks = vcat(fill(k, N), 1)
    end

    Ω = tt_randn(rng, dims, rks, normalization=normalization, orthogonal=orthogonal, right=right, T=T)

    if orthogonal
        for i = 1:N
            Ω.ttv_vec[i] *= sqrt(dims[i]*rks[i+1]/rks[i])
        end
    else
        for i = 1:N
            Ω.ttv_vec[i] /= sqrt(rks[i])
        end
    end

    for i in 1:batch
        if batch == 1 # Julia doesn't like assignments like gtt[:,1] = scalar
            gtt = tt_dot(X[1], Ω)
        else
            gtt[:,i] = tt_dot(X[i], Ω)
        end
    end

    return gtt, Ω
end

function recursive_sketch(rng::AbstractRNG, k::Int64, dims::NTuple{N,Int64}; T::Type{<:Number}=Float64) where {N}
    Ω = 1
    for i in 1:N
        if i == 1
            M = randn(rng, T, k, dims[i])/sqrt(k)
        else
            #M = randn(rng, T, k, k*dims[i])/sqrt(k)
            M = transpose(khatri_rao(randn(rng, T, k, k), randn(rng, T, dims[i], k)/sqrt(k)))
        end
        I1 = Matrix{T}(I, dims[i], dims[i])
        Ω = M*kron(Ω, I1)
    end
    return Ω
end

function khatri_rao(A::AbstractMatrix, B::AbstractMatrix)
    if size(A, 2) != size(B, 2)
        throw(ArgumentError("Matrices must have the same number of columns for Khatri-Rao product"))
    end
    N = size(A, 2)
    # Determine the size of the resulting columns
    col_size = size(A, 1) * size(B, 1)
    # Preallocate the result matrix
    result = Matrix{eltype(A)}(undef, col_size, N)
    for j in 1:N
        result[:, j] = kron(A[:, j], B[:, j])
    end
    return result
end