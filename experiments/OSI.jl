using TensorTrains
using LinearAlgebra
using Plots
using Statistics
include("utilities.jl")
include("sketches.jl")

# Number of independent runs
num_realizations = 100

#Random.seed!(2)
rng = MersenneTwister(2)

# Ranks and physical dimensions of tensors
ranks = [1, 2, 5, 10, 20]
rank_X = 10
d = 2 # dimensions
N = 10 # cores

# embedding dimensions
k = r -> 2*r

λ_max = 20 #Subspace dimension
k_max = k(λ_max) #Embedding dimension

# Tensor train and embedding dimension
I = ntuple(i -> d, N)


### Need to find a better sample subspace
#X = cell(λ_max,1)
X = SizedArray{Tuple{λ_max}, TTvector{Float64,N}}(undef)


# [V,~] = qr(randn([prod(I),lambda_max]), "econ")
# 
# for r=1:λ_max
#     Xr = tt_tensor(reshape(V(:,r), I))
#     X[r] = Xr/norm(Xr)
#     X[r] = core2cell(X[r])
# end

for λ = 1:λ_max
    tmp1 = vcat(1, fill(rank_X, N-1), 1)
    tmp2 = [reverse(cumprod(reverse(I)))..., 1]
    R = [min(tmp1[i], tmp2[i]) for i in eachindex(tmp1)]
    X_λ = tt_randn(rng, I, R, orthogonal=true)
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
gtt = Vector{Array{Float64,3}}(undef, λ_max)
ogtt = Vector{Array{Float64,3}}(undef, λ_max)
ftt = Vector{Array{ComplexF64,3}}(undef, length(ranks))

for λ = 1:λ_max
    gtt[λ] = zeros(k(λ), λ, num_realizations)
    ogtt[λ] = zeros(k(λ), λ, num_realizations)

    for T = 1:num_realizations
        gtt[λ][:,:,T],_ = GTT(rng, X, I, k(λ), batch=λ)
        gtt[λ][:,:,T] = gtt[λ][:,:,T]*C[1:λ, 1:λ]

        ogtt[λ][:,:,T],_ = GTT(rng, X, I, k(λ), batch=λ, orthogonal=true)
        ogtt[λ][:,:,T] = ogtt[λ][:,:,T]*C[1:λ, 1:λ]
    end
end



for R in eachindex(ranks)
    ftt[R] = zeros(ComplexF64, k_max, λ_max, num_realizations)
    for T = 1:num_realizations
        ftt[R][:,:,T],_ = TTR(rng, X, I, [ranks[R]], k_max, normalization="spherical", T=ComplexF64)
        ftt[R][:,:,T] = sqrt(k_max)*ftt[R][:,:,T]*C # correct for 1/sqrt(k) in TTR
    end
end


α_gtt = zeros(λ_max)
α_ogtt = zeros(λ_max)
α_ftt = zeros(length(ranks),λ_max,num_realizations)

β_gtt = zeros(λ_max)
β_ogtt = zeros(λ_max)
β_ftt = zeros(length(ranks), λ_max,num_realizations)

for λ = 1:λ_max
    α_gtt[λ], β_gtt[λ] = injectivity_dilation(gtt[λ], num_realizations, stats=median)
    α_ogtt[λ], β_ogtt[λ] = injectivity_dilation(ogtt[λ], num_realizations, stats=median)
end

for R = 1:length(ranks)
    for λ = 1:λ_max
        α_ftt[R,λ,:], β_ftt[R,λ,:] = injectivity_dilation(ftt[R][1:k(λ), 1:λ, :]/sqrt(k(λ)), num_realizations) #Median is computed when plotting because of memory layout
    end
end


p = plot(range(1,λ_max), α_gtt ,label="GTT", linestyle=:dash, yscale=:log10)
plot!(range(1,λ_max), α_ogtt ,label="OGTT", linestyle=:dashdot, yscale=:log10)

for R in eachindex(ranks)
    plot!(range(1,λ_max), median(α_ftt[R,:,:],dims=2), label="TT($(ranks[R]))", yscale=:log10)
end

title!("Injectivities (d = $d, N = $N)")
xlabel!("Subspace dimension")
ylabel!("Median Injectivity")
display(p)

p = plot(range(1,λ_max), β_gtt ,label="GTT", linestyle=:dash, yscale=:log10)
plot!(range(1,λ_max), β_ogtt ,label="OGTT", linestyle=:dashdot, yscale=:log10)

for R in eachindex(ranks)
    plot!(range(1,λ_max), median(β_ftt[R,:,:],dims=2), label="TT($(ranks[R]))", yscale=:log10)
end

title!("Dilation (d = $d, N = $N)")
xlabel!("Subspace dimension")
ylabel!("Median Dilation")
display(p)
