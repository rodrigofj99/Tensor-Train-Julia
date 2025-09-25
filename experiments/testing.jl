using TensorTrains
import LinearAlgebra.norm
using Plots
using Random
using Statistics
include("sketches.jl")
include("utilities.jl")


Random.seed!(1234)
d = 2
N = 2
I = ntuple(i -> d, N) #d * ones(1,N)
#R = (1..., ntuple(i->10,N-1)..., 1)
R = [1..., 10*ones(Int64, N-1)..., 1]
X = rand_tt(I,R)
X = X/norm(X)
norm_X = norm(X)

#sketch = rand_tt(ComplexF64,I,[1*ones(Int64, N);1])
sketch = tensor_rand_proj(Float64, I, 10)
#println(sketch)
#println(X)
println(tt_dot(X,sketch))

#= println(mean(I))
println(mean(R)) =#