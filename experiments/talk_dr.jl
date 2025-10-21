using TensorTrains
using LinearAlgebra
using Plots
using Statistics
include("utilities.jl")
include("sketches.jl")


# Number of independent runs
num_realizations = 10
#Random.seed!(2)
rng = MersenneTwister(2);

# Ranks and physical dimensions of tensors
ranks = [1, 2, 17]
rank_X = 10
d = 2 # dimensions
N = 50 # cores

K_max = 68
#Ks = round.(Int, logrange(1, K_max, length=300))
Ks = range(1, K_max)
dr_ttr = zeros(num_realizations, length(Ks), length(ranks))
dr_ttpr = zeros(num_realizations, length(Ks), length(ranks))
dr_gtt = zeros(num_realizations, length(Ks))


# Tensor train and embedding dimension
dims = ntuple(i -> d, N)
Rs = vcat(1, fill(10, N-1), 1)
X = tt_randn(rng, dims, Rs)
X = X/norm(X)
X_s = SizedArray{Tuple{1},TTvector{Float64,N}}(X) # wrap in SizedArray for batch size 1 (Improves performance)
norm_X = norm(X)

#= for T in 1:num_realizations
    for k in eachindex(Ks)
        for r in eachindex(ranks)
            ttr,_ = TTR(rng, X_s, dims, [ranks[r]], Ks[k], orthogonal=false, normalization="spherical", T=ComplexF64)
            dr_ttr[T,k,r] = abs(norm(ttr)^2/norm_X^2 - 1)

            ttpr,_ = TTPR(rng, X_s, dims, [ranks[r]], Ks[k] ÷ ranks[r], ranks[r], orthogonal=false, normalization="spherical", T=ComplexF64)
            dr_ttpr[T,k,r] = abs(norm(ttpr)^2/norm_X^2 - 1)
        end
        gtt,_ = GTT(rng, X_s, dims, Ks[k])
        dr_gtt[T,k] = abs(norm(gtt)^2/norm_X^2 - 1)           
    end
end =#

for T in 1:num_realizations
    for r in eachindex(ranks)
        ttr,_ = TTR(rng, X_s, dims, [ranks[r]], K_max, orthogonal=false, normalization="spherical", T=ComplexF64)
        ttr *= sqrt(K_max)       
        ttpr,_ = TTPR(rng, X_s, dims, [ranks[r]], K_max ÷ ranks[r], ranks[r], orthogonal=false, normalization="spherical", T=ComplexF64)
        ttpr *= sqrt(K_max ÷ ranks[r])

        for k in eachindex(Ks)
            dr_ttr[T,k,r] = abs(norm(ttr[1:Ks[k]]./sqrt(Ks[k]))^2 /norm_X^2 - 1)
            dr_ttpr[T,k,r] = abs(norm(ttpr[1:Ks[k]]./sqrt(Ks[k] ÷ ranks[r]))^2 /norm_X^2 - 1)
        end
    end

    for k in eachindex(Ks)
        gtt,_ = GTT(rng, X_s, dims, Ks[k])
        dr_gtt[T,k] = abs(norm(gtt)^2/norm_X^2 - 1)    
    end       
end


for i in eachindex(ds)
    p = plot(Ks, transpose(median(dr_gtt, dims=1)),label="GTT", linestyle=:dot, yscale=:log10)

    for r in eachindex(ranks)
        plot!(p, Ks, transpose(median(dr_ttr[:,:,r], dims=1)),label="TT($(ranks[r]))", yscale=:log10)
        plot!(p, Ks, transpose(median(dr_ttpr[:,:,r], dims=1)),label="TT(k/$(ranks[r]), $(ranks[r]))", yscale=:log10)
    end
    
    title!("Distortions (d = $(ds[i]), N = $(Ns[i]))")
    xlabel!("Embedding Dimension k")
    ylabel!("Distortion Ratio")
    display(p)
end
