using TensorTrains
import LinearAlgebra.norm
include("utilities.jl")


function TTR(rng::AbstractRNG,::Type{T}, dims:: NTuple{M,Int64}, rks::Int) where {T,M}
    N = length(dims)
    #X = Array{T,3}[]
    X = Vector{Array{T,3}}(undef, N)
    rks = vcat(1, fill(rks, N-1), 1)
    
    # If using orthogonal cores, implement upper bound on the ranks
    #R = [min(R, i) for i in [reverse(cumprod(reverse(I)))..., 1]];
    for n = 1 : N
        # # Try spherical distribution normalization - amazing for Khatri-Rao!
        Xn = randn(rng, T, dims[n], rks[n], rks[n + 1])
        Xn = (sqrt(dims[n]) / (rks[n]*rks[n+1])^(0.25)) ./ vecnorm(Xn,2,1) .* Xn
        #push!(X, Xn)
        X[n] = Xn    
        #TODO: Port code below to Julia
        # Orthogonal version of the spherical distribution?
        # [Xn,~] = qr(randn(R(n+1)*I(n), R(n), like=1i), 'econ');
        # Xn = reshape(Xn', [R(n),I(n),R(n+1)]);

        # # Normalizations: do they make a difference?
        # s = sqrt(I(n)*R(n+1)/R(n)); 
        # X{n} = reshape(s * Xn,[R(n),I(n),R(n+1)]);

        # X{n} = (sqrt(I(n)) / (R(n)*R(n+1))^(0.25)) ./ vecnorm(Xn,2,2) .* Xn;

        # Or pure Gaussian
        # X{n} = (1 / (R(n)*R(n+1))^(0.25)) .* randn(R(n), I(n), R(n + 1), like=1i);
    end

    return TTvector{T,M}(N, X, dims, rks, zeros(Int64, N))
end