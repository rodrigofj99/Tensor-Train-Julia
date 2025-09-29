using TensorTrains
using LinearAlgebra
using Plots
using Statistics
include("utilities.jl")
include("sketches.jl")


# Number of independent runs
num_realizations = 100
#Random.seed!(2)
rng = MersenneTwister(2);

# Ranks and physical dimensions of tensors
ranks = [1, 2, 5, 10]
rank_X = 10
ds = 2#[10, 7, 2] # dimensions
Ns = 10#[3, 7, 10] # cores

K_max = 10
#Ks = round.(Int, logrange(1, K_max, length=300))
Ks = range(1, K_max)
#dr_gaussian = zeros(length(ds), num_realizations, length(Ks))
dr_ttr = zeros(length(ds), num_realizations, length(Ks), length(ranks))
dr_gtt = zeros(length(ds), num_realizations, length(Ks))
dr_ogtt = zeros(length(ds), num_realizations, length(Ks))

#time_gaussian = zeros(length(ds), num_realizations, length(Ks))
time_ttr = zeros(length(ds), num_realizations, length(Ks), length(ranks))
time_gtt = zeros(length(ds), num_realizations, length(Ks))
time_ogtt = zeros(length(ds), num_realizations, length(Ks))


for p in eachindex(ds)
    # Tensor train and embedding dimension
    d = ds[p]
    N = Ns[p]
    I = ntuple(i -> d, N)
    Rs = vcat(1, fill(10, N-1), 1)
    X = tt_randn(rng, I, Rs)
    X = X/norm(X)
    norm_X = norm(X)
    Xfull = vec(full(X))

    for T in 1:num_realizations
        for k in eachindex(Ks)
            for r in eachindex(ranks)
                time = @elapsed begin
                    ttr,_ = TTR(rng, [X], I, [ranks[r]], Ks[k], orthogonal=false, normalization="spherical", T=ComplexF64)
                end
                time_ttr[p,T,k,r] = time
                dr_ttr[p,T,k,r] = abs(norm(ttr)^2/norm_X^2 - 1)

            end
            #= time = @elapsed begin
                gaussian_rp = 1/sqrt(k)*randn(rng,k,prod(I)) * Xfull
            end
            time_gaussian[p,T,k] = time
            dr_gaussian[p,T,k] = abs(norm(gaussian_rp)^2/norm_X^2 - 1)     =#
    
            time = @elapsed begin
                gtt,_ = GTT(rng, [X], I, Ks[k])
            end
            time_gtt[p,T,k] = time
            dr_gtt[p,T,k] = abs(norm(gtt)^2/norm_X^2 - 1)
    
            time = @elapsed begin
                ogtt,_ = GTT(rng, [X], I, Ks[k], orthogonal=true)
            end
            time_ogtt[p,T,k] = time
            dr_ogtt[p,T,k] = abs(norm(ogtt)^2/norm_X^2 - 1)
            
        end
    end
end


#= for i in eachindex(ds)
    #p = plot(Ks, transpose(median(time_gaussian[i,:,:], dims=1)), label="Gaussian RP", linestyle=:dash, yscale=:log10) #mean gives a 1xK_max array
    p = plot(Ks, transpose(median(time_gtt[i,:,:], dims=1)),label="GTT", linestyle=:dot, yscale=:log10)
    plot!(Ks, transpose(median(time_ogtt[i,:,:], dims=1)),label="OGTT", linestyle=:dashdot, yscale=:log10)

    for r in eachindex(ranks)
        plot!(Ks, transpose(median(time_ttr[i,:,:,r], dims=1)),label="TT($(ranks[r]))", yscale=:log10)
    end
    
    title!("Distortions (d = $(ds[i]), N = $(Ns[i]))")
    xlabel!("Embedding Dimension")
    ylabel!("Time")
    display(p)
end =#



for i in eachindex(ds)
    #p = plot(Ks, transpose(median(dr_gaussian[i,:,:], dims=1)), label="Gaussian RP", linestyle=:dash, yscale=:log10) #mean gives a 1xK_max array
    p = plot(Ks, transpose(median(dr_gtt[i,:,:], dims=1)),label="GTT", linestyle=:dot, yscale=:log10)
    plot!(Ks, transpose(median(dr_ogtt[i,:,:], dims=1)),label="OGTT", linestyle=:dashdot, yscale=:log10)

    for r in eachindex(ranks)
        plot!(Ks, transpose(median(dr_ttr[i,:,:,r], dims=1)),label="TT($(ranks[r]))", yscale=:log10)
    end
    
    title!("Distortions (d = $(ds[i]), N = $(Ns[i]))")
    xlabel!("Embedding Dimension")
    ylabel!("Distortion Ratio")
    display(p)
end
