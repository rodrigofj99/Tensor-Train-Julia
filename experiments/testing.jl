using TensorTrains
import LinearAlgebra.norm
using Plots
using Random
using Statistics
include("sketches.jl")
include("utilities.jl")



function compare_tt_cores(A::TTvector, B::TTvector)
    @assert A.N == B.N
    for k = 1:A.N
        sa = size(A.ttv_vec[k]); sb = size(B.ttv_vec[k])
        println("core $k sizes: ", sa, "  vs  ", sb)
        @assert sa == sb "core sizes differ at $k!"
        diff = abs.(A.ttv_vec[k] .- B.ttv_vec[k])
        maxabs = maximum(diff)
        denom = maximum(abs.(A.ttv_vec[k])) + eps(eltype(diff))
        maxrel = maxabs/denom
        println("  max abs diff = ", maxabs, "   max rel diff = ", maxrel)
        println("  first few entries A vs B:")
        println(Array(A.ttv_vec[k][1:min(3,sa[1]), 1:min(2,sa[2]), 1:min(2,sa[3])]))
        println(Array(B.ttv_vec[k][1:min(3,sb[1]), 1:min(2,sb[2]), 1:min(2,sb[3])]))
    end
end


rng = MersenneTwister(1234)   # fixed seed for reproducibility


λ_max = 5 #Subspace dimension
N = 10  # number of cores
d = 2   # physical dimension of each core
I = ntuple(i -> d, N)
# produce single sketch with old function (TTR1) and new function (TTR)
X = Vector{TTvector{Float64,N}}(undef, λ_max)

for λ = 1:λ_max
    tmp1 = vcat(1, fill(10, N-1), 1)
    tmp2 = [reverse(cumprod(reverse(I)))..., 1]
    R = [min(tmp1[i], tmp2[i]) for i in eachindex(tmp1)]
    X_λ = tt_randn(rng, I, R, orthogonal=true)
    X[λ] = X_λ/norm(X_λ)
end

for λ = 1:λ_max
rng = MersenneTwister(1234)   # fixed seed for reproducibility

    sketch_old = TTR1(rng, ComplexF64, I, Rs[2])    # one TTvector
    rng = MersenneTwister(1234)   # fixed seed for reproducibility

    ttr_new, Ω_new = TTR(rng, X[1], I, Rs[2], 1; normalization="spherical", T=ComplexF64)
    sketch_new = Ω_new[1]

    v_old = tt_dot(X[λ], sketch_old)
    v_new = tt_dot(X[λ], sketch_new)

    println("tt_dot old = ", v_old)
    println("tt_dot new = ", ttr_new[1])
    println("tt_dot new inside = ", v_new)
    println("ratio new/old = ", v_new / v_old,"\n\n")
end

#compare_tt_cores(sketch_old, sketch_new)

# Simple mutation test
orig = copy(sketch_new.ttv_vec[1])
println("old sketch core[1][1,1,1] = ", sketch_old.ttv_vec[1][1,1,1])
sketch_new.ttv_vec[1][1,1,1] += 1.0 + 0im
println("old sketch core[1][1,1,1] = ", sketch_old.ttv_vec[1][1,1,1])
println("new sketch core[1][1,1,1] = ", sketch_new.ttv_vec[1][1,1,1])
# revert
sketch_new.ttv_vec[1][1,1,1] = orig[1,1,1]

