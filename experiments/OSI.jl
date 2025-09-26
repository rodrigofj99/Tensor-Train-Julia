using TensorTrains
using LinearAlgebra
using Plots
using Statistics
include("utilities.jl")
include("sketches.jl")


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
X = Vector{TTvector{Float64,N}}(undef, λ_max)


# [V,~] = qr(randn([prod(I),lambda_max]), "econ")
# 
# for r=1:λ_max
#     Xr = tt_tensor(reshape(V(:,r), I))
#     X[r] = Xr/norm(Xr)
#     X[r] = core2cell(X[r])
# end

for λ = 1:λ_max
    tmp1 = vcat(1, fill(rank_X, N-1), 1)
    #tmp2 = vcat(reverse(cumprod(reverse(I))), 1)
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

# Number of independent runs
num_realizations = 100

# Collect all the sketches for further post-processing
gtt = Vector{Array{Float64,3}}(undef, λ_max)
ogtt = Vector{Array{Float64,3}}(undef, λ_max)

for λ = 1:λ_max

    gtt[λ] = zeros(k(λ), λ, num_realizations)
    ogtt[λ] = zeros(k(λ), λ, num_realizations)

    for T = 1:num_realizations
        R = vcat(fill(k(λ), N), 1)
        sketch = tt_randn(rng, I, R)
        for s = 1:λ
            gtt[λ][:,s,T] = tt_dot(X[s], sketch)
        end
        gtt[λ][:,:,T] = gtt[λ][:,:,T]*C[1:λ, 1:λ]
        
        #sketch = tt_randn_orth_cores(I, k(λ), "left")
        R = [min(k(λ), i) for i in [reverse(cumprod(reverse(I)))..., 1]]
        sketch = tt_randn(rng, I, R, orthogonal=true)
        for s = 1:λ
            ogtt[λ][:,s,T] = tt_dot(X[s], sketch)
        end
        ogtt[λ][:,:,T] = ogtt[λ][:,:,T]*C[1:λ, 1:λ]
    end
end

ftt = Vector{Array{ComplexF64,3}}(undef, length(ranks))

for R in eachindex(ranks)
    ftt[R] = zeros(ComplexF64, k_max, λ_max, num_realizations)
    for T = 1:num_realizations
        for i = 1:k_max
            sketch = TTR(rng, ComplexF64, I, ranks[R])
            for λ = 1:λ_max
                ftt[R][i,λ,T] = tt_dot(X[λ], sketch)
            end
        end
        ftt[R][:,:,T] = ftt[R][:,:,T]*C
    end
end

alpha_gtt = zeros(λ_max, num_realizations)
alpha_ogtt = zeros(λ_max, num_realizations)

beta_gtt = zeros(λ_max, num_realizations)
beta_ogtt = zeros(λ_max, num_realizations)


for r = 1:λ_max
    for T = 1:num_realizations
        alpha_gtt[r,T] = svdvals(gtt[r][:,:,T])[end]^2
        alpha_ogtt[r,T] = svdvals(ogtt[r][:,:,T])[end]^2

        beta_gtt[r,T] = svdvals(gtt[r][:,:,T])[1]^2
        beta_ogtt[r,T] = svdvals(ogtt[r][:,:,T])[1]^2
    end
end


alpha_ftt = zeros(length(ranks),λ_max, num_realizations)
beta_ftt = zeros(length(ranks), λ_max, num_realizations)

for R = 1:length(ranks)
    for T = 1:num_realizations
        for λ = 1:λ_max
            alpha_ftt[R,λ,T] = svdvals(ftt[R][1:k(λ), 1:λ,T]/sqrt(k(λ)))[end]^2
            beta_ftt[R,λ,T] = svdvals(ftt[R][1:k(λ), 1:λ,T]/sqrt(k(λ)))[1]^2
        end
    end
end


p = plot(range(1,λ_max), median(alpha_gtt, dims=2) ,label="GTT", linestyle=:dash, yscale=:log10)
plot!(range(1,λ_max), median(alpha_ogtt, dims=2) ,label="OGTT", linestyle=:dashdot, yscale=:log10)

for R in eachindex(ranks)
    plot!(range(1,λ_max), median(alpha_ftt[R,:,:], dims=2) ,label="TT($(ranks[R]))", yscale=:log10)
end

title!("Injectivities (d = $d, N = $N)")
xlabel!("Subspace dimension")
ylabel!("Median Injectivity")
display(p)

p = plot(range(1,λ_max), median(beta_gtt, dims=2) ,label="GTT", linestyle=:dash, yscale=:log10)
plot!(range(1,λ_max), median(beta_ogtt, dims=2) ,label="OGTT", linestyle=:dashdot, yscale=:log10)

for R in eachindex(ranks)
    plot!(range(1,λ_max), median(beta_ftt[R,:,:], dims=2) ,label="TT($(ranks[R]))", yscale=:log10)
end

title!("Dilation (d = $d, N = $N)")
xlabel!("Subspace dimension")
ylabel!("Median Dilation")
display(p)
