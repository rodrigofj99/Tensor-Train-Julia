using TensorTrains
import LinearAlgebra.norm
using StaticArrays
include("utilities.jl")


function TTR(rng::AbstractRNG, X::SizedArray{Tuple{M},TTvector{T0,N}}, dims::NTuple{N,Int64}, rk::Vector{Int64}, k::Int64; batch::Int64=M, orthogonal=false, normalization::AbstractString, T::Type{<:Number}=Float64) where {T0,N,M}
    Ω = Vector{TTvector{T,N}}(undef, k)
    ttr = zeros(T, k, batch)

    rks = length(rk)==1 ? vcat(1, fill(rk[1], N-1), 1) : rk

    for i in 1:k
        cores = Vector{Array{T,3}}(undef, N)
        for j = 1 : N
            if orthogonal
                tmp = [reverse(cumprod(reverse(dims)))..., 1]
                rks = [min(rks[i], tmp[i]) for i in eachindex(tmp)]
                core_j = ortho_randn(rng, dims[j], rks[j], rks[j+1], right=true, T=T)
            else
                core_j = randn(rng, T, dims[j], rks[j], rks[j+1])
            end

            if normalization == "spherical"
                core_j = (sqrt(dims[j]) / (rks[j]*rks[j+1])^(0.25)) ./ vecnorm(core_j,2,1) .* core_j
            elseif normalization == "gaussian"
                core_j = (1 / (rks[j]*rks[j+1])^(0.25)) .* core_j
            elseif normalization == "orthogonal"
                core_j *= sqrt(dims[j]*rks[j+1]/rks[j])
            else
                error("Normalization not recognized")
            end
    
            cores[j] = core_j    
        end

        ω = TTvector{T,N}(N, cores, dims, rks, zeros(Int64, N))
        Ω[i] = ω

        for j = 1:batch
            ttr[i,j] = tt_dot(X[j], ω)/sqrt(k)
        end       
    end
    return ttr, Ω
end


function GTT(rng::AbstractRNG, X::SizedArray{Tuple{M},TTvector{T0,N}}, dims::NTuple{N,Int64}, k::Int64; batch::Int64=M, orthogonal=false, T::Type{<:Number}=Float64) where {T0,N,M}
    rks = vcat(fill(k, N), 1)
    gtt = zeros(T, k, batch)
    if orthogonal
        rks = [min(k, i) for i in [reverse(cumprod(reverse(dims)))..., 1]]
        Ω = tt_randn(rng, dims, rks, orthogonal=true, right=true, T=T)
    else
        Ω = tt_randn(rng, dims, rks, T=T)
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