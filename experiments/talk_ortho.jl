using TensorTrains
using LinearAlgebra
using Plots
using Statistics
using LaTeXStrings
include("utilities.jl")
include("sketches.jl")

# Number of independent runs
num_realizations = 10

#Random.seed!(2)
rng = MersenneTwister(2)

# embedding dimensions
k = r -> 3*r+8 # embedding dimension as a linear function of rank

λ_max = 10 #Subspace dimension
k_max = k(λ_max) #Embedding dimension

# Ranks and physical dimensions of tensors
ranks = [1, 12] #[1, 2, 17] #divisors of k_max
ranks_X = [10]
d = 2 # dimensions
N = 5 # cores

# Tensor train and embedding dimension
dims = ntuple(i -> d, N)

X = SizedArray{Tuple{λ_max}, TTvector{Float64,N}}(undef)

for rX in eachindex(ranks_X)
    for λ = 1:λ_max
        tmp1 = vcat(1, fill(ranks_X[rX], N-1), 1)
        tmp2 = [reverse(cumprod(reverse(dims)))..., 1]
        R = [min(tmp1[i], tmp2[i]) for i in eachindex(tmp1)]
        X_λ = tt_randn(rng, dims, R, normalization="none", orthogonal=true)
        if rX == 2
            X_λ = (1 + rand()*10^(-2))*X_λ
        end
        X[λ] = X_λ/norm(X_λ)
    end

    A = zeros(λ_max, λ_max)
    for λ = 1:λ_max
        A[λ,λ] = tt_dot(X[λ],X[λ])
        for j = λ+1:λ_max
            A[λ,j] = tt_dot(X[λ],X[j])
            A[j,λ] = conj(A[λ,j])
        end
    end

    # This factor orthogonalizes the space: X*C is orthogonal. We sketch X but
    # look to compute Omega'*X*C
    C = inv(cholesky(A).U)

    # Collect all the sketches for further post-processing
    gtt_type = Float64 #ComplexF64
    ftt_type = ComplexF64
    ttpr_type = ComplexF64

    gtt = Vector{Array{gtt_type,3}}(undef, λ_max)
    ftt = Vector{Array{ftt_type,3}}(undef, length(ranks))
    ttpr = Vector{Array{ftt_type,3}}(undef, length(ranks))

    gtt2 = Vector{Array{gtt_type,3}}(undef, λ_max)
    ftt2 = Vector{Array{ftt_type,3}}(undef, length(ranks))
    ttpr2 = Vector{Array{ftt_type,3}}(undef, length(ranks))

    for λ = 1:λ_max
        gtt[λ] = zeros(gtt_type, k(λ), λ, num_realizations)
        gtt2[λ] = zeros(gtt_type, k(λ), λ, num_realizations)

        for T = 1:num_realizations
            gtt[λ][:,:,T],_ = GTT(rng, X, dims, k(λ), normalization="none", batch=λ, T=gtt_type)
            gtt[λ][:,:,T] = gtt[λ][:,:,T]*C[1:λ, 1:λ]

            gtt2[λ][:,:,T],_ = GTT(rng, X, dims, k(λ), normalization="none", orthogonal=true, batch=λ, T=gtt_type)
            gtt2[λ][:,:,T] = gtt2[λ][:,:,T]*C[1:λ, 1:λ]
        end
    end



    for R in eachindex(ranks)
        ftt[R] = zeros(ftt_type, k_max, λ_max, num_realizations)
        ttpr[R] = zeros(ttpr_type, k_max, λ_max, num_realizations)

        ftt2[R] = zeros(ftt_type, k_max, λ_max, num_realizations)
        ttpr2[R] = zeros(ttpr_type, k_max, λ_max, num_realizations)

        for T = 1:num_realizations
            ftt[R][:,:,T],_ = TTR(rng, X, dims, [ranks[R]], k_max, normalization="spherical", T=ftt_type)
            ftt[R][:,:,T] = sqrt(k_max)*ftt[R][:,:,T]*C # correct for 1/sqrt(k) in TTR

            ttpr[R][:,:,T],_ = TTPR(rng, X, dims, [ranks[R]], k_max ÷ ranks[R], ranks[R], normalization="spherical", T=ttpr_type)
            ttpr[R][:,:,T] = sqrt(k_max ÷ ranks[R])*ttpr[R][:,:,T]*C # correct for 1/sqrt(k) in TTPR

            
            ftt2[R][:,:,T],_ = TTR(rng, X, dims, [ranks[R]], k_max, normalization="spherical", orthogonal=true, T=ftt_type)
            ftt2[R][:,:,T] = sqrt(k_max)*ftt2[R][:,:,T]*C # correct for 1/sqrt(k) in TTR

            ttpr2[R][:,:,T],_ = TTPR(rng, X, dims, [ranks[R]], k_max ÷ ranks[R], ranks[R], normalization="spherical", orthogonal=true, T=ttpr_type)
            ttpr2[R][:,:,T] = sqrt(k_max ÷ ranks[R])*ttpr2[R][:,:,T]*C # correct for 1/sqrt(k) in TTPR
        end
    end


    α_gtt = zeros(λ_max)
    α_ftt = zeros(length(ranks),λ_max,num_realizations)
    α_ttpr = zeros(length(ranks),λ_max,num_realizations)

    α_gtt2 = zeros(λ_max)
    α_ftt2 = zeros(length(ranks),λ_max,num_realizations)
    α_ttpr2 = zeros(length(ranks),λ_max,num_realizations)


    β_gtt = zeros(λ_max)
    β_ftt = zeros(length(ranks), λ_max,num_realizations)
    β_ttpr = zeros(length(ranks), λ_max,num_realizations)

    β_gtt2 = zeros(λ_max)
    β_ftt2 = zeros(length(ranks), λ_max,num_realizations)
    β_ttpr2 = zeros(length(ranks), λ_max,num_realizations)

    for λ = 1:λ_max
        α_gtt[λ], β_gtt[λ] = injectivity_dilation(gtt[λ], num_realizations, stats=median)
        α_gtt2[λ], β_gtt2[λ] = injectivity_dilation(gtt2[λ], num_realizations, stats=median)
    end

    for R in eachindex(ranks)
        for λ = 1:λ_max
            α_ftt[R,λ,:], β_ftt[R,λ,:] = injectivity_dilation(ftt[R][1:k(λ), 1:λ, :]/sqrt(k(λ)), num_realizations) #Median is computed when plotting because of memory layout
            α_ttpr[R,λ,:], β_ttpr[R,λ,:] = injectivity_dilation(ttpr[R][1:k(λ), 1:λ, :]/sqrt(k(λ) ÷ ranks[R]), num_realizations) #Median is computed when plotting because of memory layout
           
            α_ftt2[R,λ,:], β_ftt2[R,λ,:] = injectivity_dilation(ftt2[R][1:k(λ), 1:λ, :]/sqrt(k(λ)), num_realizations) #Median is computed when plotting because of memory layout
            α_ttpr2[R,λ,:], β_ttpr2[R,λ,:] = injectivity_dilation(ttpr2[R][1:k(λ), 1:λ, :]/sqrt(k(λ) ÷ ranks[R]), num_realizations) #Median is computed when plotting because of memory layout
        end
    end

    p1 = plot(range(1,λ_max), α_gtt, label="GTT", yscale=:log10, linestyle=:dot, seriescolor=:black, legend=:topright)
    plot!(p1, range(1,λ_max), α_gtt2, label="OGTT", yscale=:log10, linestyle=:dot, seriescolor=:red)

    plot!(p1, range(1,λ_max), median(α_ftt[1,:,:],dims=2), label="TT($(ranks[1]))", yscale=:log10, linestyle=:dash)
    plot!(p1, range(1,λ_max), median(α_ftt2[1,:,:],dims=2), label="OTT($(ranks[1]))", yscale=:log10, linestyle=:dash)

    plot!(p1, range(1,λ_max), median(α_ttpr[2,:,:],dims=2), label="TT(k(λ)/$(ranks[2]), $(ranks[2]))", yscale=:log10, linestyle=:solid)
    plot!(p1, range(1,λ_max), median(α_ttpr2[2,:,:],dims=2), label="OTT(k(λ)/$(ranks[2]), $(ranks[2]))", yscale=:log10, linestyle=:solid)
    
    xlabel!(p1, "Subspace dimension λ")
    ylabel!(p1, "Median Injectivity  " * L"\sigma^2_{\min}")
    display(p1)

    p2 = plot(range(1,λ_max), β_gtt, label="GTT", yscale=:log10, linestyle=:dot, seriescolor=:black, legend=:bottomright)
    plot!(p2, range(1,λ_max), β_gtt2, label="OGTT", yscale=:log10, linestyle=:dot, seriescolor=:red)

    plot!(p2, range(1,λ_max), median(β_ftt[1,:,:],dims=2), label="TT($(ranks[1]))", yscale=:log10, linestyle=:dash)
    plot!(p2, range(1,λ_max), median(β_ftt2[1,:,:],dims=2), label="OTT($(ranks[1]))", yscale=:log10, linestyle=:dash)

    plot!(p2, range(1,λ_max), median(β_ttpr[2,:,:],dims=2), label="TT($(k_max/ranks[2]), $(ranks[2]))", yscale=:log10, linestyle=:solid)
    plot!(p2, range(1,λ_max), median(β_ttpr2[2,:,:],dims=2), label="OTT($(k_max/ranks[2]), $(ranks[2]))", yscale=:log10, linestyle=:solid)

    xlabel!(p2, "Subspace dimension λ")
    ylabel!(p2, "Median Dilation  " * L"\sigma^2_{\max}")
    display(p2)

    #= if rX == 2
        titles =  "Test vectors of rank $(ranks_X[rX])+perturbation in " * L"%$(d)^{%$(N)}"
    else
       titles =  "Test vectors of rank $(ranks_X[rX]) in " * L"%$(d)^{%$(N)}"
    end

    p3 = plot(p1, p2, layout=(1,2), plot_title=titles)  
    display(p3) =#
end
