using TensorTrains
using LinearAlgebra
using StaticArrays
include("utilities.jl")


function TTPR(rng::AbstractRNG, X::SizedArray{Tuple{M},TTvector{T0,N}}, dims::NTuple{N,Int64}, rk::Vector{Int64}, P::Int64, R::Int64; normalization::AbstractString="none", batch::Int64=M, orthogonal=false, right=true, T::Type{<:Number}=Float64) where {T0,N,M}
    k = Int(R*P)
    #println(R, "*",P,"=",k)
    #k = Int(rk[1]*P)
    Ω = Vector{TTvector{T,N}}(undef, k)
    ttr = zeros(T, k, batch)

    rks = length(rk)==1 ? vcat(R, fill(rk[1], N-1), 1) : rk
    #rks = length(rk)==1 ? vcat(fill(rk[1], N), 1) : rk

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
            #if rk[1] == 1 # Julia doesn't like assignments like ttr[:,1] = scalar
            if R == 1 # Julia doesn't like assignments like ttr[:,1] = scalar
                #ttr[i,j] = partial_contraction(X[j], ω)[1]/sqrt(P)
                ttr[i,j] = tt_dot(X[j], ω)/sqrt(P)
            else
                #ttr[(i-1)*rk[1]+1:i*rk[1], j] = partial_contraction(X[j], ω)[1]/sqrt(P)
                #ttr[(i-1)*rk[1]+1:i*rk[1], j] = tt_dot(X[j], ω)/sqrt(P)
                ttr[(i-1)*R+1:i*R, j] = tt_dot(X[j], ω)/sqrt(P)
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
            #ttr[i,j] = partial_contraction(X[j], ω)[1][1]/sqrt(k)
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
            #gtt = partial_contraction(X[1], Ω)[1]
            gtt = tt_dot(X[1], Ω)
        else
            #gtt[:,i] = partial_contraction(X[i], Ω)[1]
            gtt[:,i] = tt_dot(X[i], Ω)
        end
    end

    return gtt, Ω
end

function recursive_sketch(rng::AbstractRNG, k::Int64, dims::NTuple{N,Int64}; T::Type{<:Number}=Float64) where {N}
    Ω = 1
    rks = vcat(fill(k, N), 1)

    #= for i in 1:N
        ω_i = randn(rng, T, dims[i]*rks[i+1], rks[i])/sqrt(rks[i])
        t = prod(dims[1:i-1])
        Ii = Matrix{T}(I, t, t)
        Ω = kron(Ii, ω_i)*Ω      
    end =#

    t = ones(Int64, N+1)
    t[2:end] .= cumprod(dims)

    for i in 1:N
        ω_i = randn(rng, T, rks[i], dims[i]*rks[i+1])/sqrt(rks[i])
        Ii = Matrix{T}(I, t[i], t[i])
        Ω = Ω*kron(Ii, ω_i)    
    end

    return Ω
end




function recursive_kron(A::TTvector{T1,N}, B::TTvector{T2,N}) where {T1<:Number,T2<:Number,N}
    if T1 == ComplexF64 || T2 == ComplexF64
        T = ComplexF64
    else
        T = Float64
    end

    dims = A.ttv_dims
    rks = A.ttv_rks
    dks = B.ttv_rks

    #W = reshape(A.ttv_vec[N], rks[N], dims[N]*rks[N+1])*reshape(B.ttv_vec[N], dims[N]*dks[N+1], dks[N])
    W = reshape(permutedims(A.ttv_vec[N], (2,1,3)), rks[N], dims[N]*rks[N+1])*reshape(permutedims(B.ttv_vec[N], (1,3,2)), dims[N]*dks[N+1], dks[N])

    for i = N-1:-1:1
        t1 = reshape(permutedims(A.ttv_vec[i], (2,1,3)), rks[i], dims[i]*rks[i+1])
        t2 = reshape(permutedims(B.ttv_vec[i], (1,3,2)), dims[i]*dks[i+1], dks[i])
        Ii = Matrix{T}(I, dims[i], dims[i])
        W = t1*kron(Ii, W)*t2
    end

    return W
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




function build_Omega_from_B(rng::AbstractRNG, k::Int64, d::NTuple{N,Int64}) where {N}
    R = vcat(fill(k, N), 1)

    # --- 1) generate Gaussian B-cores: B[k] has shape (R[k], d[k], R[k+1]) ---
    B = Vector{Array{Float64,3}}(undef, N)
    σ = 1/sqrt(R[i])
    for k in 1:N
        B[k] = σ * randn(rng, R[k], d[k], R[k+1])
    end

    # --- 2) prefix products of physical dims: D0 = 1, Dk = prod_{j<=k} d[j] ---
    Dprefix = ones(Int, N+1)
    for k in 1:N
        Dprefix[k+1] = Dprefix[k] * d[k]
    end
    # Dprefix[k] = D_{k-1} in the notation above

    # --- 3) build ω_k = I_{D_{k-1}} ⊗ M_k for each k ---
    omegas = Vector{Matrix{Float64}}(undef, N)
    for k in 1:N
        Rk   = R[k]
        Rkn  = R[k+1]
        dk   = d[k]
        # permute B[k] from (α, i, β) to (α, β, i) so columns become (i,β) with β fastest
        Bperm = permutedims(B[k], (1,3,2))               # shape (Rk, Rkn, dk)
        # reshape to matrix M_k of shape (Rk, d_k * R_{k+1})
        M_k = reshape(Bperm, Rk, dk * Rkn)               # M_k[α, (i-1)*Rkn + β] = B[α,i,β]
        Ileft = Matrix{Float64}(I, Dprefix[k], Dprefix[k]) # explicit identity of size D_{k-1}
        omegas[k] = kron(Ileft, M_k)                      # shape (D_{k-1}*Rk) × (D_k*R_{k+1})
    end

    # --- 4) multiply ω_1 * ω_2 * ... * ω_N to form Omega ---
    Omega = omegas[1]
    for k in 2:N
        Omega = Omega * omegas[k]
    end

    return Omega, B
end