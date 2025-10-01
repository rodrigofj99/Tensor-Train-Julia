using TensorTrains
import LinearAlgebra.norm
using Plots
using Random
using Statistics
include("sketches.jl")
include("utilities.jl")


rng = MersenneTwister(4321)   # fixed seed for reproducibility
N = 10  # number of cores
d = 2   # physical dimension of each core
dims = ntuple(i -> d, N)
X = tt_randn(rng, dims, vcat(1, fill(10, N-1), 1))

rng = MersenneTwister(1234)  # reset seed for reproducibility
SX, Ω1 = GTT(rng, SizedArray{Tuple{1},TTvector{Float64,N}}(X), dims, 10) # random TT to be sketched
SX = transpose(SX) # make it a column vector

rng = MersenneTwister(1234)   # reset seed for reproducibility
Ω2 = recursive_sketch(rng, 10, dims) 
SX2 = Ω2 * vec(ttv_to_tensor(X))
SX3 = Ω2 * vec(full(X))

println(SX,"\n\n")
println(SX2,"\n\n")
println(SX3,"\n\n")

println(SX ./ SX2, "\n\n")
println(SX ./ SX3)