using TensorTrains
import LinearAlgebra.norm
using Plots
using Random
using Statistics
include("sketches.jl")
include("utilities.jl")


rng = MersenneTwister(4321)   # fixed seed for reproducibility
N = 2  # number of cores
d = 2   # physical dimension of each core
dims = ntuple(i -> d, N)
rks = vcat(1, fill(9, N-1), 1)
k = 5
X = tt_randn(rng, dims, rks)



rng = MersenneTwister(1234)  # reset seed for reproducibility      
dks = vcat(fill(k, N), 1)
B = tt_randn(rng, dims, dks)
for i = 1:N
    B.ttv_vec[i] /= sqrt(dks[i])
end
        #= t1 = reshape(X.ttv_vec[1], rks[1], dims[1]*rks[2])
        t2 = reshape(B.ttv_vec[1], dims[1]*R[2], R[1])
        println(t1*t2) =#



PC = partial_contraction(X,B)[1]
println(PC)

RK = recursive_kron(X,B)
println(RK)



rng = MersenneTwister(1234)  # reset seed for reproducibility
SX, Ω1 = GTT(rng, SizedArray{Tuple{1},TTvector{Float64,N}}(X), dims, k) # random TT to be sketched

println("size of SX: ",size(SX))
println("size of PC: ",size(PC))

rng = MersenneTwister(1234)   # reset seed for reproducibility
Ω2 = recursive_sketch(rng, k, dims) 
SX2 = Ω2 * vec(ttv_to_tensor(X))

println(SX)
println(SX2)