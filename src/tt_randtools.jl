using TensorOperations
using LinearAlgebra

"""
    default_rank_heuristic(y::TTvector{T,N}) -> Vector{Int}
    default_rank_heuristic(Y::Vector{TTvector{T,N}}) -> Vector{Int}

Compute heuristic sketch ranks for randomized rounding algorithms.

Returns the input ranks with an oversampling factor of 50% of the maximum rank added
to internal ranks (boundary ranks remain 1). This provides numerical stability for
randomized QR-based rounding methods.

For a vector of TTvectors, returns the element-wise maximum of input ranks plus oversampling,
useful for randomized rounding of linear combinations α₁*y₁ + α₂*y₂ + ... + αₘ*yₘ.

# Arguments
- `y::TTvector`: Input TTvector
- `Y::Vector{TTvector}`: Vector of TTvectors (for linear combinations)

# Returns
- `rks::Vector{Int}`: Suggested sketch ranks with oversampling (length N+1)

# Formula
- oversampling = ⌈0.5 × max(ranks)⌉
- rks[1] = 1 (boundary)
- rks[k] = input_rks[k] + oversampling for k = 2:N
- rks[N+1] = 1 (boundary)
"""
function default_rank_heuristic(y::TTvector{T,N}) where {T,N}
  rks = deepcopy(y.ttv_rks)
  oversampling = ceil(Int, .5*maximum(rks))
  for k=2:N
    rks[k] += oversampling
  end
  return rks
end

function default_rank_heuristic(Y::Vector{TTvector{T,N}}) where {T,N}
  rks = deepcopy(Y[1].ttv_rks)
  for j=2:length(Y)
    rks = max.(rks, Y[j].ttv_rks)
  end
  oversampling = ceil(Int, .5*maximum(rks))
  for k=2:N
    rks[k] += oversampling
  end
  return rks
end

"""
    ttrand_rounding(y::TTvector{T,N}, rks=default_rank_heuristic(y); orthogonal=true, seed=1234) -> TTvector{T,N}
    ttrand_rounding(A::TToperator{T,N}, y::TTvector{T,N}, b::TTvector{T,N}, rks=default_rank_heuristic(y); orthogonal=true, seed=1234) -> TTvector{T,N}
    ttrand_rounding(α::Vector{T}, y::Vector{TTvector{T,N}}, rks=default_rank_heuristic(y); orthogonal=true, seed=1234) -> TTvector{T,N}
    ttrand_rounding(α::Vector{T}, A::TToperator{T,N}, y::Vector{TTvector{T,N}}, rks=default_rank_heuristic(y); orthogonal=true, seed=1234) -> TTvector{T,N}

Randomized TT rounding using sketching and orthogonalization.

Compresses a TTvector (or combination of TTvectors) to smaller ranks using random sketching
followed by QR factorization with column pivoting. More efficient than SVD-based rounding for
large-scale problems.

# Variants
1. Single TTvector: rounds `y`
2. Residual: rounds `A*y - b` without explicitly forming the product (useful for iterative solvers)
3. Linear combination: rounds `α₁*y₁ + α₂*y₂ + ... + αₘ*yₘ`
4. Mixed combination: rounds `α₁*A*y₁ + α₂*y₂ + ... + αₘ*yₘ` (useful for gradient descent or Krylov solvers)

# Arguments
- `y::TTvector{T,N}`: Input TTvector to round
- `A::TToperator{T,N}`: Operator (for residual or mixed combinations)
- `b::TTvector{T,N}`: Right-hand side vector (for residual)
- `α::Vector{T}`: Coefficients (for linear combinations)
- `y::Vector{TTvector{T,N}}`: Vector of TTvectors (for linear combinations)
- `rks::Vector{Int}=default_rank_heuristic(...)`: Target ranks (with oversampling for stability)

# Keyword Arguments
- `orthogonal::Bool=true`: Use orthogonal random sketches (recommended for stability)
- `seed::Int=1234`: Random seed for reproducibility

# Returns
- `result::TTvector{T,N}`: Rounded TTvector with left-orthogonal cores

# Algorithm
Implements "Randomize then Orthogonalize" approach from:
Al Daas, Ballard, Cazeaux, Hallman, Miedlar, Pasha, Reid & Saibaba (2023),
"Randomized algorithms for rounding in the tensor-train format",
SIAM J. Sci. Comput., 45(1), pp. A74–A95, https://doi.org/10.1137/21M1451191

1. Generate recursive sketch using tt_recursive_sketch
2. Perform left-to-right sweep with randomized QR decomposition
3. Truncate ranks based on numerical rank from column-pivoted QR

# Notes
- Resulting TTvector has left-orthogonal cores (ot[k] = 1)
- Actual ranks may be smaller than input rks due to numerical rank detection
- More efficient than tt_rounding for low-to-moderate accuracy requirements
- For combinations, sketches each term separately then combines during QR sweep for efficiency

# Example
```julia
# Simple rounding
y_rounded = ttrand_rounding(y, rks)

# Residual for iterative solver
residual = ttrand_rounding(A, x, b, rks)

# Linear combination
sum = ttrand_rounding([0.5, 0.5], [y1, y2], rks)

# Gradient descent step: x_new = x - α*∇f(x)
x_new = ttrand_rounding([1.0, -step_size], A, [x, gradient], rks)
```
"""
function ttrand_rounding(y::TTvector{T,N}, rks=default_rank_heuristic(y); orthogonal=true, seed=1234) where {T,N}
  dims = y.ttv_dims
  vec = Vector{Array{T,3}}(undef, N)
  W, sketch_rks = tt_recursive_sketch(T, y, rks; orthogonal=orthogonal, reverse=true, seed=seed)
  ot = zeros(Int, N)

  # Randomized sketching and orthogonalization
  yₖ = reshape(y.ttv_vec[1], dims[1], 1, y.ttv_rks[2])
  @inbounds begin
    for k in 1:N-1
      # Randomized QR decomposition
      Zₖ = zeros(T, dims[k],rks[k],sketch_rks[k+1])
      @tensoropt (αₖ₊₁,ρₖ,ρₖ₊₁)  Zₖ[iₖ,ρₖ,ρₖ₊₁] = yₖ[iₖ,ρₖ,αₖ₊₁]*W[k+1][αₖ₊₁,ρₖ₊₁]
      Zₖ = reshape(Zₖ,dims[k]*rks[k],sketch_rks[k+1])
      Q, _ = qr!(Zₖ, ColumnNorm())
      Q = Matrix(Q)
      rks[k+1] = size(Q,2)
      vec[k] = reshape(Q,dims[k],rks[k],rks[k+1])
      ot[k] = 1

      #update left parts
      yₖ₊₁ = zeros(T, dims[k+1],rks[k+1],y.ttv_rks[k+2])
      @tensoropt (αₖ₊₁,αₖ₊₂,ρₖ₊₁)  yₖ₊₁[iₖ₊₁,ρₖ₊₁,αₖ₊₂] = yₖ[iₖ,ρₖ,αₖ₊₁]*vec[k][iₖ,ρₖ,ρₖ₊₁]*y.ttv_vec[k+1][iₖ₊₁,αₖ₊₁,αₖ₊₂]
      yₖ = yₖ₊₁
    end
    rks[N+1] = 1
    vec[N] = reshape(yₖ, dims[N],rks[N],rks[N+1])
  end

  return TTvector{T,N}(N,vec,dims,rks,ot)
end

"""
    ttrand_rounding(y::TTvector{T,N}, rmax::Int; orthogonal=true, seed=1234) -> TTvector{T,N}

Convenience variant of ttrand_rounding that takes a maximum rank as second argument.
Generates uniform ranks [1, rmax, rmax, ..., rmax, 1] for sketching.
"""
function ttrand_rounding(y::TTvector{T,N}, rmax::Int; orthogonal::Bool=true, seed::Int=1234) where {T,N}
    # Generate uniform ranks with rmax
    rks = ones(Int, N+1)
    rks[2:N] .= rmax
    return ttrand_rounding(y, rks; orthogonal=orthogonal, seed=seed)
end

function ttrand_rounding(Atto::TToperator{T,N}, y::TTvector{T,N}, b::TTvector{T,N}, rks=default_rank_heuristic(y); orthogonal=true, seed=1234) where {T,N}
  dims = y.ttv_dims
  vec = Vector{Array{T,3}}(undef, N)
  WAy, sketch_rks = tt_recursive_sketch(T, Atto, y, rks; orthogonal=orthogonal, reverse=true, seed=seed)
  Wb,  sketch_rks = tt_recursive_sketch(T,       b, rks; orthogonal=orthogonal, reverse=true, seed=seed)
  ot = zeros(Int, N)

# Randomized sketching and orthogonalization
  Ayₖ = zeros(T, dims[1], 1, y.ttv_rks[2], Atto.tto_rks[2])
  @tensoropt (αₖ,βₖ,αₖ₊₁,βₖ₊₁) Ayₖ[iₖ,ρₖ,αₖ₊₁,βₖ₊₁] = y.ttv_vec[1][jₖ,αₖ,αₖ₊₁]*Atto.tto_vec[1][iₖ,jₖ,βₖ,βₖ₊₁]
  bₖ = reshape(b.ttv_vec[1], dims[1], 1, b.ttv_rks[2])
  @inbounds begin
    for k in 1:N-1
      # Randomized QR decomposition
      Zₖ = zeros(T, dims[k],rks[k],sketch_rks[k+1])
      @tensoropt (αₖ₊₁,βₖ₊₁,ρₖ,ρₖ₊₁)  Zₖ[iₖ,ρₖ,ρₖ₊₁] := Ayₖ[iₖ,ρₖ,αₖ₊₁,βₖ₊₁]*WAy[k+1][αₖ₊₁,βₖ₊₁,ρₖ₊₁] - bₖ[iₖ,ρₖ,αₖ₊₁]*Wb[k+1][αₖ₊₁,ρₖ₊₁]
      Zₖ = reshape(Zₖ,dims[k]*rks[k],sketch_rks[k+1])
      Q, _ = qr!(Zₖ, ColumnNorm())
      Q = Matrix(Q)
      rks[k+1] = size(Q,2)
      vec[k] = reshape(Q,dims[k],rks[k],rks[k+1])
      ot[k] = 1

      #update left parts
      Ayₖ₊₁ = zeros(T, dims[k+1],rks[k+1],y.ttv_rks[k+2],Atto.tto_rks[k+2])
       bₖ₊₁ = zeros(T, dims[k+1],rks[k+1],b.ttv_rks[k+2])
      @tensoropt (αₖ₊₁,βₖ₊₁,αₖ₊₂,βₖ₊₂,ρₖ₊₁) Ayₖ₊₁[iₖ₊₁,ρₖ₊₁,αₖ₊₂,βₖ₊₂] = Ayₖ[iₖ,ρₖ,αₖ₊₁,βₖ₊₁]*vec[k][iₖ,ρₖ,ρₖ₊₁]*y.ttv_vec[k+1][jₖ₊₁,αₖ₊₁,αₖ₊₂]*Atto.tto_vec[k+1][iₖ₊₁,jₖ₊₁,βₖ₊₁,βₖ₊₂]
      @tensoropt (αₖ₊₁,     αₖ₊₂,    ρₖ₊₁)  bₖ₊₁[iₖ₊₁,ρₖ₊₁,αₖ₊₂     ] =  bₖ[iₖ,ρₖ,αₖ₊₁    ]*vec[k][iₖ,ρₖ,ρₖ₊₁]*b.ttv_vec[k+1][iₖ₊₁,αₖ₊₁,αₖ₊₂]
      Ayₖ = Ayₖ₊₁
      bₖ  = bₖ₊₁
    end
    rks[N+1] = 1
    vec[N] = reshape(Ayₖ,dims[N],rks[N],rks[N+1]) - reshape(bₖ,dims[N],rks[N],rks[N+1])
  end

  return TTvector{T,N}(N,vec,dims,rks,ot)
end

function ttrand_rounding(α::Vector{T}, y::Vector{TTvector{T,N}}, rks=default_rank_heuristic(y); orthogonal=true, seed=1234) where {T,N}
  m = length(α)
  dims = y[1].ttv_dims
  @assert length(y) == m && all(y[j].ttv_dims == dims for j=2:m)

  W = Vector{Vector{Array{T,2}}}(undef, m)
  sketch_rks = Vector{Vector{Int64}}(undef, m)

  for j=1:m
    W[j], sketch_rks[j] = tt_recursive_sketch(T, y[j], rks; orthogonal=orthogonal, reverse=true, seed=seed)
    @assert sketch_rks[j] == sketch_rks[1]
  end
  sketch_rks = sketch_rks[1]

  vec = Vector{Array{T,3}}(undef, N)
  ot = zeros(Int, N)

  # Randomized sketching and orthogonalization
  Yₖ = [α[j].*reshape(y[j].ttv_vec[1], dims[1], 1, y[j].ttv_rks[2]) for j=1:m]
  @inbounds begin
    for k in 1:N-1
      # Randomized QR decomposition
      Zₖ = zeros(T, dims[k],rks[k],sketch_rks[k+1])
      for j=1:m
        @tensoropt (αₖ₊₁,ρₖ,ρₖ₊₁) Zₖ[iₖ,ρₖ,ρₖ₊₁] += Yₖ[j][iₖ,ρₖ,αₖ₊₁]*W[j][k+1][αₖ₊₁,ρₖ₊₁]
      end
      Zₖ = reshape(Zₖ,dims[k]*rks[k],sketch_rks[k+1])
      Q, _ = qr!(Zₖ, ColumnNorm())
      rks[k+1] = size(Q,2)
      vec[k] = reshape(Matrix(Q),dims[k],rks[k],rks[k+1])
      ot[k] = 1

      #update left parts
      Yₖ₊₁ = [ zeros(T, dims[k+1],rks[k+1],y[j].ttv_rks[k+2]) for j=1:m ]
      for j=1:m
        @tensoropt (αₖ₊₁,αₖ₊₂,ρₖ₊₁)  Yₖ₊₁[j][iₖ₊₁,ρₖ₊₁,αₖ₊₂] = Yₖ[j][iₖ,ρₖ,αₖ₊₁]*vec[k][iₖ,ρₖ,ρₖ₊₁]*y[j].ttv_vec[k+1][iₖ₊₁,αₖ₊₁,αₖ₊₂]
      end
      Yₖ = Yₖ₊₁
    end
    rks[N+1] = 1
    vec[N] = sum(Yₖ[j] for j=1:m)
  end

  return TTvector{T,N}(N,vec,dims,rks,ot)
end


function ttrand_rounding(α::Vector{T}, A::TToperator{T,N}, y::Vector{TTvector{T,N}}, rks=default_rank_heuristic(y); orthogonal=true, seed=1234) where {T,N}
  m = length(α)
  dims = y[1].ttv_dims
  @assert length(y) == m && all(y[j].ttv_dims == dims for j=2:m)

  W = Vector{Vector{Array{T,2}}}(undef, m)

  W₁, sketch_rks = tt_recursive_sketch(T, A, y[1], rks; orthogonal=orthogonal, reverse=true, seed=seed)
  W[1] = [reshape(W₁[k], y[1].ttv_rks[k]*A.tto_rks[k], sketch_rks[k]) for k=1:N+1]
  for j=2:m
    Wⱼ, sketchj_rks = tt_recursive_sketch(T, y[j], rks; orthogonal=orthogonal, reverse=true, seed=seed)
    W[j] = Wⱼ
    @assert sketchj_rks == sketch_rks
  end

  vec = Vector{Array{T,3}}(undef, N)
  ot = zeros(Int, N)

  # Randomized sketching and orthogonalization
  Yₖ = Vector{Array{T,3}}(undef, m)

  Ay₁ = zeros(T, dims[1], 1, y[1].ttv_rks[2], A.tto_rks[2])
  @tensoropt (α₂,β₂) Ay₁[i₁,ρₖ,α₂,β₂] = y[1].ttv_vec[1][j₁,ρₖ,α₂]*A.tto_vec[1][i₁,j₁,ρₖ,β₂]
  Ay₁ .*= α[1]
  Yₖ[1] = reshape(Ay₁, dims[1], 1, y[1].ttv_rks[2]*A.tto_rks[2])
  for j=2:m
    Yₖ[j] = α[j].*reshape(y[j].ttv_vec[1], dims[1], 1, y[j].ttv_rks[2])
  end
  @inbounds begin
    for k in 1:N-1
      # Randomized QR decomposition
      Zₖ = zeros(T, dims[k],rks[k],sketch_rks[k+1])
      for j=1:m
        @tensoropt (αₖ₊₁,ρₖ,ρₖ₊₁) Zₖ[iₖ,ρₖ,ρₖ₊₁] += Yₖ[j][iₖ,ρₖ,αₖ₊₁]*W[j][k+1][αₖ₊₁,ρₖ₊₁]
      end
      Zₖ = reshape(Zₖ,dims[k]*rks[k],sketch_rks[k+1])
      Q, _ = qr!(Zₖ, ColumnNorm())
      rks[k+1] = size(Q,2)
      vec[k] = reshape(Matrix(Q),dims[k],rks[k],rks[k+1])
      ot[k] = 1

      #update left parts
      Yₖ₊₁ = Vector{Array{T,3}}(undef, m)

      y₁ₖ = reshape(Yₖ[1], dims[k], rks[k], y[1].ttv_rks[k+1], A.tto_rks[k+1])
      Ay₁ₖ₊₁ = zeros(T, dims[k+1], rks[k+1], y[1].ttv_rks[k+2], A.tto_rks[k+2])
      @tensoropt (ρₖ,ρₖ₊₁,αₖ₊₁,βₖ₊₁,αₖ₊₂,βₖ₊₂) Ay₁ₖ₊₁[iₖ₊₁,ρₖ₊₁,αₖ₊₂,βₖ₊₂] = y₁ₖ[iₖ,ρₖ,αₖ₊₁,βₖ₊₁]*vec[k][iₖ,ρₖ,ρₖ₊₁]*y[1].ttv_vec[k+1][jₖ₊₁,αₖ₊₁,αₖ₊₂]*A.tto_vec[k+1][iₖ₊₁,jₖ₊₁,βₖ₊₁,βₖ₊₂]
      Yₖ₊₁[1] = reshape(Ay₁ₖ₊₁, dims[k+1], rks[k+1], y[1].ttv_rks[k+2]*A.tto_rks[k+2])
      for j=2:m
        Yₖ₊₁[j] = zeros(T, dims[k+1],rks[k+1],y[j].ttv_rks[k+2])
        @tensoropt (ρₖ₊₁,αₖ₊₁,αₖ₊₂)  Yₖ₊₁[j][iₖ₊₁,ρₖ₊₁,αₖ₊₂] = Yₖ[j][iₖ,ρₖ,αₖ₊₁]*vec[k][iₖ,ρₖ,ρₖ₊₁]*y[j].ttv_vec[k+1][iₖ₊₁,αₖ₊₁,αₖ₊₂]
      end
      Yₖ = Yₖ₊₁
    end
    rks[N+1] = 1
    vec[N] = sum(Yₖ[j] for j=1:m)
  end

  return TTvector{T,N}(N,vec,dims,rks,ot)
end



function dot_randrounding(A::TToperator,x::TTvector)
  y = A*x
  y = ttrand_rounding(y)
  return tt_rounding(y;tol=1e-8)
end

"""
returns a stable solution to A\\B
"""
function stable_inverse(A;ε=1e-12)
  u,s,v = svd(A)
  return v[:,s.>maximum(s)*ε]*Diagonal(1 ./s[s.>maximum(s)*ε])*u[:,s.>maximum(s)*ε]'
end

"""
    stta(y::TTvector{T,N}; rks=default_rank_heuristic(y), 
         seed_left=1234, seed_right=5678, orthogonal=true) -> TTvector{T,N}

Streaming Tensor Train Approximation (STTA) - approximate a TTvector using two-sided random sketches.

Constructs a low-rank approximation by computing left and right random sketches, forming
overlap matrices, and solving local systems. Unlike standard rounding methods, STTA can
leverage structure in the input tensor without accessing the full tensor cores.

# Arguments
- `y::TTvector{T,N}`: Input TTvector to approximate

# Keyword Arguments
- `rks::Vector{Int}=default_rank_heuristic(y)`: Target ranks for approximation
- `seed_left::Int=1234`: Random seed for left sketch
- `seed_right::Int=5678`: Random seed for right sketch
- `orthogonal::Bool=true`: Use orthogonal random sketches

# Returns
- `y_approx::TTvector{T,N}`: Approximated TTvector with ranks determined adaptively

# Algorithm
Based on:
Kressner, Vandereycken & Voorhaar (2022), "Streaming Tensor Train Approximation",
SIAM J. Sci. Comput., 45(5), pp. A2610–A2629, https://arxiv.org/abs/2208.02600

1. Generate left and right sketches via stta_sketch (automatically applies 50% oversampling to left ranks)
2. Compute overlap matrices Ω[k] and sketched cores Ψ[k] from the sketches
3. Solve local systems adaptively: either Ψ[k] * Ω[k]⁻¹ or Ω[k]⁻¹ * Ψ[k+1]
4. Return approximation with adaptively determined ranks

# Notes
- Uses stable_inverse with SVD truncation (ε=1e-12) for numerical stability
- Adapts sweep direction at each site based on sketch dimensions
- stta_sketch automatically applies 50% oversampling to left ranks for optimal performance
- More efficient than SVD-based methods for structured or streaming data
"""
function stta(y_tt::TTvector{T,N}; rks=default_rank_heuristic(y_tt), 
              seed_left::Int=1234, seed_right::Int=5678, orthogonal::Bool=true) where {T,N}
  # stta_sketch handles the oversampling internally
  Ω,Ψ = stta_sketch(y_tt, rks; seed_left=seed_left, seed_right=seed_right, orthogonal=orthogonal)
  result_rks = ones(Int, N+1)
  # Deep copy Ψ to avoid reference issues
  Ψ_copy = [copy(Ψ[k]) for k=1:N]
  for k in 1:N-1
    if size(Ω[k],1)<size(Ω[k],2) #rR>rL
      Ψ_temp = reshape(Ψ_copy[k],:,size(Ψ_copy[k],3))*stable_inverse(Ω[k])
      Ψ_copy[k]= reshape(Ψ_temp,size(Ψ_copy[k],1),size(Ψ_copy[k],2),:)
      result_rks[k+1] = size(Ψ_copy[k],3)
    else 
      Ψ_temp = stable_inverse(Ω[k])*reshape(permutedims(Ψ_copy[k+1],(2,1,3)),size(Ψ_copy[k+1],2),:)
      Ψ_copy[k+1]= permutedims(reshape(Ψ_temp,:,size(Ψ_copy[k+1],1),size(Ψ_copy[k+1],3)),(2,1,3))
      result_rks[k+1] = size(Ψ_copy[k+1],2)
    end
  end
  
  # Set boundary ranks to 1 (should already be correct from sketch)
  result_rks[1] = 1
  result_rks[N+1] = 1
  return TTvector{T,N}(N,Ψ_copy,y_tt.ttv_dims,result_rks,zeros(N))
end

"""
Wrong algorithm
"""
function tt_hmt(y_tt::TTvector{T,N};rks=y_tt.ttv_rks,rmax=maximum(rks),ℓ=round(Int,maximum(rks))) where {T,N}
  rks = r_and_d_to_rks(rks.+ℓ,y_tt.ttv_dims;rmax=rmax+ℓ)
  Ω = [randn(T, y_tt.ttv_rks[k], rks[k]) for k=2:N+1]
  x_tt = zeros_tt(y_tt.ttv_dims,rks)
  y_temp = zeros(maximum(y_tt.ttv_dims),maximum(rks),maximum(y_tt.ttv_rks))
  y_temp[1:y_tt.ttv_dims[1],1:rks[1],1:y_tt.ttv_rks[2]] = copy(y_tt.ttv_vec[1])
  for k in 1:N-1
    @tensor A[iₖ,αₖ₋₁,βₖ] := @view(y_temp[1:y_tt.ttv_dims[k],1:rks[k],1:y_tt.ttv_rks[k+1]])[iₖ,αₖ₋₁,αₖ]*Ω[k][αₖ,βₖ]
    A = reshape(A,y_tt.ttv_dims[k]*rks[k],:)
    q,_ = qr!(A, ColumnNorm())
    x_tt.ttv_vec[k] = reshape(Matrix(q),y_tt.ttv_dims[k],rks[k],:)
    rks[k+1] = size(x_tt.ttv_vec[k],3)
    R_temp = q'[1:rks[k+1],:]*reshape(y_temp[1:y_tt.ttv_dims[k],1:rks[k],1:y_tt.ttv_rks[k+1]],y_tt.ttv_dims[k]*rks[k],:) #size rks[k+1] × y_tt.ttv_rks[k+1]
    @tensor (y_temp[1:y_tt.ttv_dims[k+1],1:rks[k+1],1:y_tt.ttv_rks[k+2]])[iₖ₊₁,αₖ,αₖ₊₁] = R_temp[αₖ,βₖ]*y_tt.ttv_vec[k+1][iₖ₊₁,βₖ,αₖ₊₁]
  end
  x_tt.ttv_vec[N] = y_temp[1:y_tt.ttv_dims[N],1:rks[N],1:1]
  return TTvector{T,N}(N,x_tt.ttv_vec,y_tt.ttv_dims,rks,zeros(N))
end