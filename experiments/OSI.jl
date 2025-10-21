using TensorTrains
using LinearAlgebra
using Plots
using Statistics
include("utilities.jl")
include("sketches.jl")

# Number of independent runs
num_realizations = 10

#Random.seed!(2)
rng = MersenneTwister(2)

# embedding dimensions
k = r -> 3*r+8 # embedding dimension as a linear function of rank

λ_max = 100 #Subspace dimension
k_max = k(λ_max) #Embedding dimension

# Ranks and physical dimensions of tensors
ranks = [1, 2, 14] #divisors of k_max
rank_X = 10
d = 2 # dimensions
N = 10 # cores

# Tensor train and embedding dimension
dims = ntuple(i -> d, N)


### Need to find a better sample subspace
#X = cell(λ_max,1)
X = SizedArray{Tuple{λ_max}, TTvector{Float64,N}}(undef)


# [V,~] = qr(randn([prod(dims),lambda_max]), "econ")
# 
# for r=1:λ_max
#     Xr = tt_tensor(reshape(V(:,r), dims))
#     X[r] = Xr/norm(Xr)
#     X[r] = core2cell(X[r])
# end

for λ = 1:λ_max
    tmp1 = vcat(1, fill(rank_X, N-1), 1)
    tmp2 = [reverse(cumprod(reverse(dims)))..., 1]
    R = [min(tmp1[i], tmp2[i]) for i in eachindex(tmp1)]
    X_λ = tt_randn(rng, dims, R, normalization="none", orthogonal=true)
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
ogtt_type = Float64 #ComplexF64
ftt_type = ComplexF64
oftt_type = ComplexF64
ttpr_type = ComplexF64
ottpr_type = ComplexF64

gtt = Vector{Array{gtt_type,3}}(undef, λ_max)
ogtt = Vector{Array{ogtt_type,3}}(undef, λ_max)
ftt = Vector{Array{ftt_type,3}}(undef, length(ranks))
oftt = Vector{Array{oftt_type,3}}(undef, length(ranks))
ttpr = Vector{Array{ftt_type,3}}(undef, length(ranks))
ottpr = Vector{Array{oftt_type,3}}(undef, length(ranks))

for λ = 1:λ_max
    gtt[λ] = zeros(gtt_type, k(λ), λ, num_realizations)
    ogtt[λ] = zeros(ogtt_type, k(λ), λ, num_realizations)

    for T = 1:num_realizations
        #gtt[λ][:,:,T],_ = GTT(rng, X, dims, k(λ), normalization="none", batch=λ, T=gtt_type)
        #gtt[λ][:,:,T] = gtt[λ][:,:,T]*C[1:λ, 1:λ]
        ogtt[λ][:,:,T],_ = GTT(rng, X, dims, k(λ), normalization="none", batch=λ, orthogonal=true, right=true, T=ogtt_type)
        ogtt[λ][:,:,T] = ogtt[λ][:,:,T]*C[1:λ, 1:λ]
    end
end



for R in eachindex(ranks)
    ftt[R] = zeros(ftt_type, k_max, λ_max, num_realizations)
    oftt[R] = zeros(oftt_type, k_max, λ_max, num_realizations)
    ttpr[R] = zeros(ttpr_type, k_max, λ_max, num_realizations)
    ottpr[R] = zeros(ottpr_type, k_max, λ_max, num_realizations)

    for T = 1:num_realizations
        #ftt[R][:,:,T],_ = TTR(rng, X, dims, [ranks[R]], k_max, normalization="spherical", T=ftt_type)
        #ftt[R][:,:,T] = sqrt(k_max)*ftt[R][:,:,T]*C # correct for 1/sqrt(k) in TTR

        oftt[R][:,:,T],_ = TTR(rng, X, dims, [ranks[R]], k_max, normalization="spherical", orthogonal=true, T=oftt_type)
        oftt[R][:,:,T] = sqrt(k_max)*oftt[R][:,:,T]*C # correct for 1/sqrt(k) in TTR

        #ttpr[R][:,:,T],_ = TTPR(rng, X, dims, [ranks[R]], k_max ÷ ranks[R], ranks[R], normalization="spherical", T=ttpr_type)
        #ttpr[R][:,:,T] = sqrt(k_max ÷ ranks[R])*ttpr[R][:,:,T]*C # correct for 1/sqrt(k) in TTPR

        ottpr[R][:,:,T],_ = TTPR(rng, X, dims, [ranks[R]], k_max ÷ ranks[R], ranks[R], normalization="spherical", orthogonal=true, T=ottpr_type)
        ottpr[R][:,:,T] = sqrt(k_max ÷ ranks[R])*ottpr[R][:,:,T]*C # correct for 1/sqrt(k) in TTPR
    end
end


α_gtt = zeros(λ_max)
α_ogtt = zeros(λ_max)
α_ftt = zeros(length(ranks),λ_max,num_realizations)
α_oftt = zeros(length(ranks),λ_max,num_realizations)
α_ttpr = zeros(length(ranks),λ_max,num_realizations)
α_ottpr = zeros(length(ranks),λ_max,num_realizations)

β_gtt = zeros(λ_max)
β_ogtt = zeros(λ_max)
β_ftt = zeros(length(ranks), λ_max,num_realizations)
β_oftt = zeros(length(ranks), λ_max,num_realizations)
β_ttpr = zeros(length(ranks), λ_max,num_realizations)
β_ottpr = zeros(length(ranks), λ_max,num_realizations)

for λ = 1:λ_max
    #α_gtt[λ], β_gtt[λ] = injectivity_dilation(gtt[λ], num_realizations, stats=median)
    #α_ogtt[λ], β_ogtt[λ] = injectivity_dilation(ogtt[λ], num_realizations, stats=median)
end

for R in eachindex(ranks)
    for λ = 1:λ_max
        #α_ftt[R,λ,:], β_ftt[R,λ,:] = injectivity_dilation(ftt[R][1:k(λ), 1:λ, :]/sqrt(k(λ)), num_realizations) #Median is computed when plotting because of memory layout
        α_oftt[R,λ,:], β_oftt[R,λ,:] = injectivity_dilation(oftt[R][1:k(λ), 1:λ, :]/sqrt(k(λ)), num_realizations) #Median is computed when plotting because of memory layout

        #α_ttpr[R,λ,:], β_ttpr[R,λ,:] = injectivity_dilation(ttpr[R][1:k(λ), 1:λ, :]/sqrt(k(λ) ÷ ranks[R]), num_realizations) #Median is computed when plotting because of memory layout
        α_ottpr[R,λ,:], β_ottpr[R,λ,:] = injectivity_dilation(ottpr[R][1:k(λ), 1:λ, :]/sqrt(k(λ) ÷ ranks[R]), num_realizations) #Median is computed when plotting because of memory layout
    end
end

#p1 = plot(range(1,λ_max), α_gtt, label="GTT", yscale=:log10)
#plot!(p1, range(1,λ_max), α_ogtt, label="OGTT", yscale=:log10)

p1 = plot(range(1,λ_max), α_ogtt, label="OGTT", yscale=:log10)

for R in eachindex(ranks)
    #plot!(p1, range(1,λ_max), median(α_ftt[R,:,:],dims=2), label="TT($(ranks[R]))", yscale=:log10, linestyle=:dash)
    plot!(p1, range(1,λ_max), median(α_oftt[R,:,:],dims=2), label="OTT($(ranks[R]))", yscale=:log10, linestyle=:dashdot)

    #plot!(p1, range(1,λ_max), median(α_ttpr[R,:,:],dims=2), label="TTPR($(k_max/ranks[R]), $(ranks[R]))", yscale=:log10, linestyle=:dash)
    plot!(p1, range(1,λ_max), median(α_ottpr[R,:,:],dims=2), label="OTTPR($(k_max/ranks[R]), $(ranks[R]))", yscale=:log10, linestyle=:dash)
end

title!("Injectivities (d = $d, N = $N)")
xlabel!("Subspace dimension")
ylabel!("Median Injectivity")
display(p1)

#p2 = plot(range(1,λ_max), β_gtt, label="GTT", yscale=:log10)
#plot!(p2, range(1,λ_max), β_ogtt, label="OGTT", yscale=:log10)

p2 = plot(range(1,λ_max), β_ogtt, label="OGTT", yscale=:log10)

for R in eachindex(ranks)
    #plot!(p2, range(1,λ_max), median(β_ftt[R,:,:],dims=2), label="TT($(ranks[R]))", yscale=:log10, linestyle=:dash)
    plot!(p2, range(1,λ_max), median(β_oftt[R,:,:],dims=2), label="OTT($(ranks[R]))", yscale=:log10, linestyle=:dashdot)

    #plot!(p2, range(1,λ_max), median(β_ttpr[R,:,:],dims=2), label="TT($(k_max/ranks[R]), $(ranks[R]))", yscale=:log10, linestyle=:dash)
    plot!(p2, range(1,λ_max), median(β_ottpr[R,:,:],dims=2), label="OTT($(k_max/ranks[R]), $(ranks[R]))", yscale=:log10, linestyle=:dash)
end

title!("Dilation (d = $d, N = $N)")
xlabel!("Subspace dimension")
ylabel!("Median Dilation")
display(p2)
