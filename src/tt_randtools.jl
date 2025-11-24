using TensorOperations
using LinearAlgebra
using TimerOutputs

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
    ttrand_rounding(y::TTvector, rks; orthogonal=true, seed=1234, block_rks=N, timer=TimerOutput())
    ttrand_rounding(y::TTvector, rmax::Int; orthogonal=true, seed=1234, block_rks=N, timer=TimerOutput())
    ttrand_rounding(α::Vector, y::Vector{TTvector}, rks; orthogonal=true, seed=1234, block_rks=N, timer=TimerOutput())

Randomized TT rounding using recursive sketching and orthogonalization.

Compresses a TTvector (or linear combination of TTvectors) to target ranks using random
sketching followed by QR factorization. More efficient than SVD-based
rounding for large-scale problems and moderate accuracy requirements.

# Arguments
- `y::TTvector{T,N}`: Input TTvector to round
- `α::Vector`: Coefficients for linear combination α₁*y₁ + ... + αₘ*yₘ
- `y::Vector{TTvector{T,N}}`: Vector of TTvectors for linear combination
- `rks`: Target ranks (Vector{Int} of length N+1, or computed via `default_rank_heuristic`)
- `rmax::Int`: Target rank (uniform across all bonds)

# Keyword Arguments
- `orthogonal::Bool=true`: Use orthogonal random matrices (Haar/spherical, more stable)
  vs. Gaussian random matrices
- `seed::Int=1234`: Random seed for reproducibility
- `block_rks::Int=N`: Block rank for recursive sketching (controls sketch structure,
  block_rks=1 gives Khatri-Rao structure, larger values allow more general sketches)
- `timer::TimerOutput`: Timer for performance profiling

# Returns
- `TTvector{T,N}`: Rounded TTvector with left-orthogonal cores (ot[k] = 1)

# Algorithm
Implements the "Randomize-then-Orthogonalize" (RTO) approach:
1. Generate recursive sketch via right-to-left sweep (`tt_recursive_sketch`)
2. Perform left-to-right sweep with randomized QR factorization

Reference: Al Daas, Ballard, Cazeaux, Hallman, Miedlar, Pasha, Reid & Saibaba (2023),
"Randomized algorithms for rounding in the tensor-train format",
SIAM J. Sci. Comput., 45(1), pp. A74–A95. https://doi.org/10.1137/21M1451191

# Performance Notes
- Linear combination variant sketches each term separately, then combines during QR sweep
- Resulting ranks may be smaller than target due to numerical rank detection
- For high accuracy, consider SVD-based `tt_rounding`; for moderate accuracy and large
  scale, use `ttrand_rounding`

# Examples
```julia
# Round single tensor to uniform rank 10
y_rounded = ttrand_rounding(y, 10)

# Round to specific rank vector with Gaussian sketches
rks = [1, 5, 10, 15, 10, 5, 1]
y_rounded = ttrand_rounding(y, rks; orthogonal=false)

# Linear combination: 0.5*y₁ + 0.5*y₂
y_avg = ttrand_rounding([0.5, 0.5], [y1, y2], 10)

# With block ranks for faster sketching
y_rounded = ttrand_rounding(y, 10; block_rks=4)
```
"""
function ttrand_rounding(y::TTvector{T,N}, rks=default_rank_heuristic(y); orthogonal=true, seed=1234, block_rks::Int=N, timer::TimerOutput = TimerOutput()) where {T,N}
  @timeit timer "ttrand_rounding" begin
    dims = y.ttv_dims
    vec = Vector{Array{T,3}}(undef, N)
    @timeit timer "reverse_sketch" begin
      W, sketch_rks = tt_recursive_sketch(T, y, rks; orthogonal=orthogonal, reverse=true, seed=seed, block_rks=block_rks, timer=timer)
    end
    out_rks = ones(Int, N+1)
    ot = zeros(Int, N)

    # Randomized sketching and orthogonalization
    @timeit timer "orthogonalization" begin
      yₖ = reshape(y.ttv_vec[1], dims[1], 1, y.ttv_rks[2])
      @inbounds for k in 1:N-1
        # Randomized QR decomposition
      @timeit timer "Randomized QR decomposition" begin
      @timeit timer "Sketch" begin
        Zₖ = zeros(T, dims[k],out_rks[k],sketch_rks[k+1])
        @tensoropt (αₖ₊₁,ρₖ,ρₖ₊₁)  Zₖ[iₖ,ρₖ,ρₖ₊₁] = yₖ[iₖ,ρₖ,αₖ₊₁]*W[k+1][αₖ₊₁,ρₖ₊₁]
        Zₖ = reshape(Zₖ,dims[k]*out_rks[k],sketch_rks[k+1])
      end
      @timeit timer "QR" begin
        Q, _ = qr!(Zₖ)
        Q = Matrix(Q)
      end
      @timeit timer "Update core" begin
        out_rks[k+1] = size(Q,2)
        vec[k] = reshape(Q,dims[k],out_rks[k],out_rks[k+1])
        ot[k] = 1
      end
      end
      @timeit timer "Update yₖ" begin
        #update left parts
        yₖ₊₁ = zeros(T, dims[k+1],out_rks[k+1],y.ttv_rks[k+2])
        @tensoropt (αₖ₊₁,αₖ₊₂,ρₖ₊₁)  yₖ₊₁[iₖ₊₁,ρₖ₊₁,αₖ₊₂] = yₖ[iₖ,ρₖ,αₖ₊₁]*vec[k][iₖ,ρₖ,ρₖ₊₁]*y.ttv_vec[k+1][iₖ₊₁,αₖ₊₁,αₖ₊₂]
        yₖ = yₖ₊₁
      end
      end
      vec[N] = reshape(yₖ, dims[N],out_rks[N],out_rks[N+1])
    end
    return TTvector{T,N}(N,vec,dims,out_rks,ot)
  end
end

"""
    ttrand_rounding(y::TTvector{T,N}, rmax::Int; orthogonal=true, seed=1234) -> TTvector{T,N}

Convenience variant of ttrand_rounding that takes a maximum rank as second argument.
Generates uniform ranks [1, rmax, rmax, ..., rmax, 1] for sketching.
"""
function ttrand_rounding(y::TTvector{T,N}, rmax::Int; orthogonal::Bool=true, seed::Int=1234, block_rks::Int=N, timer::TimerOutput = TimerOutput()) where {T,N}
    # Generate uniform ranks with rmax
    rks = ones(Int, N+1)
    rks[2:N] .= rmax
    return ttrand_rounding(y, rks; orthogonal=orthogonal, seed=seed, block_rks=block_rks, timer=timer)
end

function ttrand_rounding(Atto::TToperator{T,N}, y::TTvector{T,N}, b::TTvector{T,N}, rks=default_rank_heuristic(y); orthogonal=true, seed=1234, block_rks::Int=N, timer::TimerOutput = TimerOutput()) where {T,N}
  @timeit timer "ttrand_rounding" begin
    dims = y.ttv_dims
    vec = Vector{Array{T,3}}(undef, N)
    @timeit timer "reverse_sketch" begin
      WAy, sketch_rks = tt_recursive_sketch(T, Atto, y, rks; orthogonal=orthogonal, reverse=true, seed=seed, block_rks=block_rks, timer=timer)
      Wb,  sketch_rks = tt_recursive_sketch(T,       b, rks; orthogonal=orthogonal, reverse=true, seed=seed, block_rks=block_rks, timer=timer)
    end
    out_rks = ones(Int, N+1)
    ot = zeros(Int, N)

# Randomized sketching and orthogonalization
    @timeit timer "orthogonalization" begin
      @timeit timer "left_sweep" begin
        Ayₖ = zeros(T, dims[1], 1, y.ttv_rks[2], Atto.tto_rks[2])
        @tensoropt (αₖ,βₖ,αₖ₊₁,βₖ₊₁) Ayₖ[iₖ,ρₖ,αₖ₊₁,βₖ₊₁] = y.ttv_vec[1][jₖ,αₖ,αₖ₊₁]*Atto.tto_vec[1][iₖ,jₖ,βₖ,βₖ₊₁]
        bₖ = reshape(b.ttv_vec[1], dims[1], 1, b.ttv_rks[2])
        @inbounds begin
          for k in 1:N-1
            # Randomized QR decomposition
            Zₖ = zeros(T, dims[k],out_rks[k],sketch_rks[k+1])
            @tensoropt (αₖ₊₁,βₖ₊₁,ρₖ,ρₖ₊₁)  Zₖ[iₖ,ρₖ,ρₖ₊₁] := Ayₖ[iₖ,ρₖ,αₖ₊₁,βₖ₊₁]*WAy[k+1][αₖ₊₁,βₖ₊₁,ρₖ₊₁] - bₖ[iₖ,ρₖ,αₖ₊₁]*Wb[k+1][αₖ₊₁,ρₖ₊₁]
            Zₖ = reshape(Zₖ,dims[k]*out_rks[k],sketch_rks[k+1])
            Q, _ = qr!(Zₖ)
            Q = Matrix(Q)
            out_rks[k+1] = size(Q,2)
            vec[k] = reshape(Q,dims[k],out_rks[k],out_rks[k+1])
            ot[k] = 1

            #update left parts
            Ayₖ₊₁ = zeros(T, dims[k+1],out_rks[k+1],y.ttv_rks[k+2],Atto.tto_rks[k+2])
             bₖ₊₁ = zeros(T, dims[k+1],out_rks[k+1],b.ttv_rks[k+2])
            @tensoropt (αₖ₊₁,βₖ₊₁,αₖ₊₂,βₖ₊₂,ρₖ₊₁) Ayₖ₊₁[iₖ₊₁,ρₖ₊₁,αₖ₊₂,βₖ₊₂] = Ayₖ[iₖ,ρₖ,αₖ₊₁,βₖ₊₁]*vec[k][iₖ,ρₖ,ρₖ₊₁]*y.ttv_vec[k+1][jₖ₊₁,αₖ₊₁,αₖ₊₂]*Atto.tto_vec[k+1][iₖ₊₁,jₖ₊₁,βₖ₊₁,βₖ₊₂]
            @tensoropt (αₖ₊₁,     αₖ₊₂,    ρₖ₊₁)  bₖ₊₁[iₖ₊₁,ρₖ₊₁,αₖ₊₂     ] =  bₖ[iₖ,ρₖ,αₖ₊₁    ]*vec[k][iₖ,ρₖ,ρₖ₊₁]*b.ttv_vec[k+1][iₖ₊₁,αₖ₊₁,αₖ₊₂]
            Ayₖ = Ayₖ₊₁
            bₖ  = bₖ₊₁
          end
          rks[N+1] = 1
          vec[N] = reshape(Ayₖ,dims[N],out_rks[N],out_rks[N+1]) - reshape(bₖ,dims[N],out_rks[N],out_rks[N+1])
        end
      end
    end

    return TTvector{T,N}(N,vec,dims,out_rks,ot)
  end
end

function ttrand_rounding(Atto::TToperator{T,N}, y::Vector{TTvector{T,N}}, b::TTvector{T,N}, rmax::Int; orthogonal::Bool=true, seed::Int=1234, block_rks::Int=N, timer::TimerOutput = TimerOutput()) where {T,N}
    # Generate uniform ranks with rmax
    rks = ones(Int, N+1)
    rks[2:N] .= rmax
    return ttrand_rounding(Atto, y, b, rks; orthogonal=orthogonal, seed=seed, block_rks=block_rks, timer=timer)
end

function ttrand_rounding(α::Vector{T}, y::Vector{TTvector{T,N}}, rks=default_rank_heuristic(y); orthogonal=true, seed=1234, block_rks::Int=N, timer::TimerOutput = TimerOutput()) where {T,N}
  @timeit timer "ttrand_rounding" begin
    m = length(α)
    dims = y[1].ttv_dims
    @assert length(y) == m && all(y[j].ttv_dims == dims for j=2:m)

    W = Vector{Vector{Array{T,2}}}(undef, m)
    sketch_rks = Vector{Vector{Int64}}(undef, m)

    @timeit timer "reverse_sketch" begin
      for j=1:m
        W[j], sketch_rks[j] = tt_recursive_sketch(T, y[j], rks; orthogonal=orthogonal, reverse=true, seed=seed, block_rks=block_rks, timer=timer)
        @assert sketch_rks[j] == sketch_rks[1]
      end
      sketch_rks = sketch_rks[1]
    end

    vec = Vector{Array{T,3}}(undef, N)
    out_rks = ones(Int, N+1)
    ot = zeros(Int, N)

    # Randomized sketching and orthogonalization
    @timeit timer "orthogonalization" begin
      Yₖ = [α[j].*reshape(y[j].ttv_vec[1], dims[1], 1, y[j].ttv_rks[2]) for j=1:m]
      @inbounds for k in 1:N-1
        # Randomized QR decomposition
      @timeit timer "Randomized QR decomposition" begin
        Zₖ = zeros(T, dims[k],out_rks[k],sketch_rks[k+1])
        for j=1:m
          @tensoropt (αₖ₊₁,ρₖ,ρₖ₊₁) Zₖ[iₖ,ρₖ,ρₖ₊₁] += Yₖ[j][iₖ,ρₖ,αₖ₊₁]*W[j][k+1][αₖ₊₁,ρₖ₊₁]
        end
        Zₖ = reshape(Zₖ,dims[k]*out_rks[k],sketch_rks[k+1])
        Q, _ = qr!(Zₖ)
        Q = Matrix(Q)
        out_rks[k+1] = size(Q,2)
        vec[k] = reshape(Q,dims[k],out_rks[k],out_rks[k+1])
        ot[k] = 1
      end
      @timeit timer "Update left contraction" begin
        #update left parts
        Yₖ₊₁ = [ zeros(T, dims[k+1],out_rks[k+1],y[j].ttv_rks[k+2]) for j=1:m ]
        for j=1:m
          @tensoropt (αₖ₊₁,αₖ₊₂,ρₖ₊₁)  Yₖ₊₁[j][iₖ₊₁,ρₖ₊₁,αₖ₊₂] = Yₖ[j][iₖ,ρₖ,αₖ₊₁]*vec[k][iₖ,ρₖ,ρₖ₊₁]*y[j].ttv_vec[k+1][iₖ₊₁,αₖ₊₁,αₖ₊₂]
        end
        Yₖ = Yₖ₊₁
      end
      end
      out_rks[N+1] = 1
      vec[N] = sum(Yₖ[j] for j=1:m)
    end

    return TTvector{T,N}(N,vec,dims,out_rks,ot)
  end
end

function ttrand_rounding(α::Vector{T}, y::Vector{TTvector{T,N}}, rmax::Int; orthogonal::Bool=true, seed::Int=1234, block_rks::Int=N, timer::TimerOutput = TimerOutput()) where {T,N}
    # Generate uniform ranks with rmax
    rks = ones(Int, N+1)
    rks[2:N] .= rmax
    return ttrand_rounding(α, y, rks; orthogonal=orthogonal, seed=seed, block_rks=block_rks, timer=timer)
end

function ttrand_rounding(α::Vector{T}, A::TToperator{T,N}, y::Vector{TTvector{T,N}}, rks=default_rank_heuristic(y); orthogonal=true, seed=1234, block_rks::Int=N, timer::TimerOutput = TimerOutput()) where {T,N}
  @timeit timer "ttrand_rounding" begin
    m = length(α)
    dims = y[1].ttv_dims
    @assert length(y) == m && all(y[j].ttv_dims == dims for j=2:m)

    W = Vector{Vector{Array{T,2}}}(undef, m)

    @timeit timer "reverse_sketch" begin
      W₁, sketch_rks = tt_recursive_sketch(T, A, y[1], rks; orthogonal=orthogonal, reverse=true, seed=seed, block_rks=block_rks, timer=timer)
      W[1] = [reshape(W₁[k], y[1].ttv_rks[k]*A.tto_rks[k], sketch_rks[k]) for k=1:N+1]
      for j=2:m
        Wⱼ, sketchj_rks = tt_recursive_sketch(T, y[j], rks; orthogonal=orthogonal, reverse=true, seed=seed, block_rks=block_rks, timer=timer)
        W[j] = Wⱼ
        @assert sketchj_rks == sketch_rks
      end
    end

    vec = Vector{Array{T,3}}(undef, N)
    ot = zeros(Int, N)

    # Randomized sketching and orthogonalization
    @timeit timer "orthogonalization" begin
      @timeit timer "left_sweep" begin
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
            Zₖ = zeros(T, dims[k],out_rks[k],sketch_rks[k+1])
            for j=1:m
              @tensoropt (αₖ₊₁,ρₖ,ρₖ₊₁) Zₖ[iₖ,ρₖ,ρₖ₊₁] += Yₖ[j][iₖ,ρₖ,αₖ₊₁]*W[j][k+1][αₖ₊₁,ρₖ₊₁]
            end
            Zₖ = reshape(Zₖ,dims[k]*out_rks[k],sketch_rks[k+1])
            Q, _ = qr!(Zₖ)
            Q = Matrix(Q)
            out_rks[k+1] = size(Q,2)
            vec[k] = reshape(Q,dims[k],out_rks[k],out_rks[k+1])
            ot[k] = 1

            #update left parts
            Yₖ₊₁ = Vector{Array{T,3}}(undef, m)

            y₁ₖ = reshape(Yₖ[1], dims[k], out_rks[k], y[1].ttv_rks[k+1], A.tto_rks[k+1])
            Ay₁ₖ₊₁ = zeros(T, dims[k+1], out_rks[k+1], y[1].ttv_rks[k+2], A.tto_rks[k+2])
            @tensoropt (ρₖ,ρₖ₊₁,αₖ₊₁,βₖ₊₁,αₖ₊₂,βₖ₊₂) Ay₁ₖ₊₁[iₖ₊₁,ρₖ₊₁,αₖ₊₂,βₖ₊₂] = y₁ₖ[iₖ,ρₖ,αₖ₊₁,βₖ₊₁]*vec[k][iₖ,ρₖ,ρₖ₊₁]*y[1].ttv_vec[k+1][jₖ₊₁,αₖ₊₁,αₖ₊₂]*A.tto_vec[k+1][iₖ₊₁,jₖ₊₁,βₖ₊₁,βₖ₊₂]
            Yₖ₊₁[1] = reshape(Ay₁ₖ₊₁, dims[k+1], out_rks[k+1], y[1].ttv_rks[k+2]*A.tto_rks[k+2])
            for j=2:m
              Yₖ₊₁[j] = zeros(T, dims[k+1],out_rks[k+1],y[j].ttv_rks[k+2])
              @tensoropt (ρₖ₊₁,αₖ₊₁,αₖ₊₂)  Yₖ₊₁[j][iₖ₊₁,ρₖ₊₁,αₖ₊₂] = Yₖ[j][iₖ,ρₖ,αₖ₊₁]*vec[k][iₖ,ρₖ,ρₖ₊₁]*y[j].ttv_vec[k+1][iₖ₊₁,αₖ₊₁,αₖ₊₂]
            end
            Yₖ = Yₖ₊₁
          end
          out_rks[N+1] = 1
          vec[N] = sum(Yₖ[j] for j=1:m)
        end
      end
    end

    return TTvector{T,N}(N,vec,dims,out_rks,ot)
  end
end

function ttrand_rounding(α::Vector{T}, A::TToperator{T,N}, y::Vector{TTvector{T,N}}, rmax::Int; orthogonal::Bool=true, seed::Int=1234, block_rks::Int=N, timer::TimerOutput = TimerOutput()) where {T,N}
    # Generate uniform ranks with rmax
    rks = ones(Int, N+1)
    rks[2:N] .= rmax
    return ttrand_rounding(α, A, y, rks; orthogonal=orthogonal, seed=seed, block_rks=block_rks, timer=timer)
end


function ttrand_rounding(y::NTuple{M,TTvector{T,N}}, rks=default_rank_heuristic(y); orthogonal=true, seed=1234, block_rks::Int=N, timer::TimerOutput = TimerOutput()) where {T,N,M}
  if M==1
    return ttrand_rounding(y[1], rks; orthogonal=orthogonal, seed=seed, block_rks=block_rks, timer=timer)
  end

  @timeit timer "ttrand_rounding" begin
    dims = y[1].ttv_dims
    vec = Vector{Array{T,3}}(undef, N)
    @timeit timer "reverse_sketch" begin
      W, sketch_rks = tt_recursive_sketch(T, y, rks; orthogonal=orthogonal, reverse=true, seed=seed, block_rks=block_rks, timer=timer)
    end
    out_rks = ones(Int, N+1)
    ot = zeros(Int, N)

    # Randomized sketching and orthogonalization
    @timeit timer "orthogonalization sweep" begin
      yₖ = broadcast(*, (reshape(y[i].ttv_vec[1], 
                              dims[1], 1, ntuple(j->( j==i ? y[i].ttv_rks[2] : 1), M)...)
                        for i=1:M)...
                    )
      yₖ = reshape(yₖ, dims[1], 1, prod(y[i].ttv_rks[2] for i=1:M))
      @inbounds for k in 1:N-1
        # Randomized QR decomposition
        @timeit timer "Randomized QR decomposition" begin
          Zₖ = zeros(T, dims[k],out_rks[k],sketch_rks[k+1])
          W_reshaped = reshape(W[k+1], prod(y[i].ttv_rks[k+1] for i=1:M), sketch_rks[k+1])
          @tensoropt (αₖ₊₁,ρₖ,ρₖ₊₁)  Zₖ[iₖ,ρₖ,ρₖ₊₁] = yₖ[iₖ,ρₖ,αₖ₊₁]*W_reshaped[αₖ₊₁,ρₖ₊₁]
          Zₖ = reshape(Zₖ,dims[k]*out_rks[k],sketch_rks[k+1])
          Q, _ = qr!(Zₖ)
          Q = Matrix(Q)

          out_rks[k+1] = size(Q,2)
          vec[k] = reshape(Q,dims[k],out_rks[k],out_rks[k+1])
          ot[k] = 1
        end
        @timeit timer "Update Qy" begin
          #update left parts
          v = ntuple(i->y[i].ttv_rks[k+1], M)
          w = ntuple(i->y[i].ttv_rks[k+2], M)
          A = ntuple(i->y[i].ttv_vec[k+1], M)
          @tensor Qy[αₖ₊₁,ρₖ₊₁] := yₖ[iₖ,ρₖ,αₖ₊₁]*vec[k][iₖ,ρₖ,ρₖ₊₁]

          if M==2
            Qy = reshape(Qy, v[1], v[2], out_rks[k+1])
            Qy = permutedims(Qy, (1,3,2))
            tmp = zeros(T, w[1], out_rks[k+1], w[2], dims[k+1])
            for i=1:dims[k+1]
              tmp_i = view(tmp,:,:,:,i)
              Ai = A[1][i,:,:]
              Bi = A[2][i,:,:]
              @tensor tmp_i[L,ρ,R] += Ai[l,L] * Qy[l,ρ,r] * Bi[r,R]
            end
            Qy = permutedims(tmp, (1,3,2,4))
          else
            # First pair, M>2
            Qy = reshape(Qy, v[1], v[2], prod(v[3:end])*out_rks[k+1])
            Qy = permutedims(Qy, (1,3,2))
            tmp = zeros(T, w[1], prod(v[3:end])*out_rks[k+1], w[2], dims[k+1])
            for i=1:dims[k+1]
              tmp_i = view(tmp,:,:,:,i)
              Ai = A[1][i,:,:]
              Bi = A[2][i,:,:]
              @tensor tmp_i[L,αρ,R] = Ai[l,L] * Qy[l,αρ,r] * Bi[r,R]
            end
            Qy = permutedims(tmp, (1,3,2,4))

            # Next pairs
            for m=3:2:M-1
              Qy = reshape(Qy, prod(w[1:m-1]), v[m], v[m+1], prod(v[m+2:end])*out_rks[k+1], dims[k+1])
              Qy = permutedims(Qy, (2,1,4,3,5))
              tmp = zeros(T, w[m], prod(w[1:m-1]), prod(v[m+2:end])*out_rks[k+1], w[m+1], dims[k+1])
              for i=1:dims[k+1]
                tmp_i = view(tmp,:,:,:,:,i)
                Qy_i = view(Qy,:,:,:,:,i)
                Ai = A[m][i,:,:]
                Bi = A[m+1][i,:,:]
                @tensor tmp_i[L,a,αρ,R] = Ai[l,L] * Qy_i[l,a,αρ,r] * Bi[r,R]
              end
              Qy = permutedims(tmp, (2,1,4,3,5))
            end

            if isodd(M) # Last core
              Qy = reshape(Qy, prod(w[1:M-1]), v[M], out_rks[k+1], dims[k+1])
              Qy = permutedims(Qy, (2,1,3,4))
              tmp = zeros(T, w[M], prod(w[1:M-1]), out_rks[k+1], dims[k+1])
              for i=1:dims[k+1]
                tmp_i = view(tmp,:,:,:,i)
                Qy_i = view(Qy,:,:,:,i)
                Ai = A[M][i,:,:]
                @tensor tmp_i[L,a,ρ] = Ai[l,L] * Qy_i[l,a,ρ]
              end
              Qy = permutedims(tmp, (2,1,3,4))
            end
          end
          Qy = reshape(Qy, w..., out_rks[k+1], dims[k+1])
          yₖ = reshape(permutedims(Qy, [M+2;M+1;1:M]), dims[k+1], out_rks[k+1], prod(w))
        end
      end
      vec[N] = reshape(yₖ, dims[N],out_rks[N],out_rks[N+1])
    end
    return TTvector{T,N}(N,vec,dims,out_rks,ot)
  end
end

function ttrand_rounding(y::NTuple{M,TTvector{T,N}}, rmax::Int; orthogonal::Bool=true, seed::Int=1234, block_rks::Int=N, timer::TimerOutput = TimerOutput()) where {T,M,N}
    # Generate uniform ranks with rmax
    rks = ones(Int, N+1)
    rks[2:N] .= rmax
    return ttrand_rounding(y, rks; orthogonal=orthogonal, seed=seed, block_rks=block_rks, timer=timer)
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
  u,s,v = svd(A, alg=LinearAlgebra.QRIteration())
  return v[:,s.>maximum(s)*ε]*Diagonal(1 ./s[s.>maximum(s)*ε])*u[:,s.>maximum(s)*ε]'
end

"""
    stta(y::TTvector, rks; seed_left=1234, seed_right=5678, orthogonal=true, block_rks=N, timer=TimerOutput())
    stta(y::TTvector, rmax::Int; seed_left=1234, seed_right=5678, orthogonal=true, block_rks=N, timer=TimerOutput())
    stta(α::Vector, y::Vector{TTvector}, rks; seed_left=1234, seed_right=5678, orthogonal=true, block_rks=N, timer=TimerOutput())

Streaming Tensor Train Approximation (STTA) using two-sided random sketches.

Constructs a low-rank approximation by computing left and right random sketches simultaneously,
then solving local systems to recover the TT cores. More efficient than one-sided sketching for
certain problems and can exploit structure without accessing full tensor cores.

# Arguments
- `y::TTvector{T,N}`: Input TTvector to approximate
- `α::Vector`: Coefficients for linear combination α₁*y₁ + ... + αₘ*yₘ
- `y::Vector{TTvector{T,N}}`: Vector of TTvectors for linear combination
- `rks`: Target ranks (Vector{Int} of length N+1, or computed via `default_rank_heuristic`)
- `rmax::Int`: Target rank (uniform across all bonds)

# Keyword Arguments
- `seed_left::Int=1234`: Random seed for left-to-right sketch
- `seed_right::Int=5678`: Random seed for right-to-left sketch (must differ from seed_left)
- `orthogonal::Bool=true`: Use orthogonal random matrices (Haar/spherical) vs. Gaussian
- `block_rks::Int=N`: Block rank for recursive sketching (controls sketch structure)
- `timer::TimerOutput`: Timer for performance profiling

# Returns
- `TTvector{T,N}`: Approximated TTvector with adaptively determined ranks

# Algorithm
Implements the Streaming TT Approximation (STTA) algorithm:
1. Generate forward (left-to-right) and backward (right-to-left) sketches via `stta_sketch`
   (automatically applies 50% oversampling to left/forward sketch ranks)
2. Compute overlap matrices Ω[k] = W_L[k]ᵀ * W_R[k] and sketched cores Ψ[k]
3. Solve local systems adaptively at each bond: choose Ψ[k] * Ω[k]⁻¹ or Ω[k]⁻¹ * Ψ[k+1]
   based on sketch dimensions (prefer overdetermined systems)
4. Return approximation with ranks determined adaptively from sketch dimensions

Reference: Kressner, Vandereycken & Voorhaar (2022), "Streaming Tensor Train Approximation",
SIAM J. Sci. Comput., 45(5), pp. A2610–A2629. https://arxiv.org/abs/2208.02600

# Performance Notes
- Uses `stable_inverse` with SVD truncation (ε=1e-12) for numerical stability
- Adaptively selects sweep direction at each bond based on sketch rank ratios
- Left sketch automatically oversampled by 50% for optimal conditioning
- Can be more efficient than `ttrand_rounding` for certain structured inputs

# Examples
```julia
# Round single tensor to uniform rank 10
y_approx = stta(y, 10)

# With different seeds for left/right sketches
y_approx = stta(y, 10; seed_left=1234, seed_right=5678)

# Linear combination: 0.5*y₁ + 0.5*y₂
y_avg = stta([0.5, 0.5], [y1, y2], 10)

# With Gaussian sketches and custom block ranks
y_approx = stta(y, 10; orthogonal=false, block_rks=4)
```
"""
function stta(y_tt::TTvector{T,N}, rks=default_rank_heuristic(y_tt);
              seed_left::Int=1234, seed_right::Int=5678, orthogonal::Bool=true, block_rks::Int=N, timer::TimerOutput = TimerOutput()) where {T,N}
  @timeit timer "stta" begin
    # stta_sketch handles the oversampling internally
    @timeit timer "stta_sketch" begin
      Ω,Ψ = stta_sketch(y_tt, rks; seed_left=seed_left, seed_right=seed_right, orthogonal=orthogonal, block_rks=block_rks, timer=timer)
    end

    result_rks = ones(Int, N+1)
    # Deep copy Ψ to avoid reference issues
    @timeit timer "rounding" begin
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
    end

    return TTvector{T,N}(N,Ψ_copy,y_tt.ttv_dims,result_rks,zeros(N))
  end
end

function stta(y::TTvector{T,N}, rmax::Int; 
              seed_left::Int=1234, seed_right::Int=5678, orthogonal::Bool=true, block_rks::Int=N, timer::TimerOutput = TimerOutput()) where {T,N}
    # Generate uniform ranks with rmax
    rks = ones(Int, N+1)
    rks[2:N] .= rmax
    return stta(y, rks; orthogonal=orthogonal, seed_left=seed_left, seed_right=seed_right, block_rks=block_rks, timer=timer)
end

function stta(α::Vector{T}, y::Vector{TTvector{T,N}}, rks=default_rank_heuristic(y_tt);
              seed_left::Int=1234, seed_right::Int=5678, orthogonal::Bool=true, block_rks::Int=N, timer::TimerOutput = TimerOutput()) where {T,N}
  @timeit timer "stta" begin
    m = length(α)
    dims = y[1].ttv_dims
    @assert length(y) == m && all(y[j].ttv_dims == dims for j=2:m)
    # stta_sketch handles the oversampling internally
    @timeit timer "stta_sketch" begin
      Ω,Ψ = stta_sketch(y[1], rks; seed_left=seed_left, seed_right=seed_right, orthogonal=orthogonal, block_rks=block_rks, timer=timer)
      Ω .*= α[1]
      Ψ .*= α[1]
      for j=2:m
        Ωj,Ψj = stta_sketch(y[j], rks; seed_left=seed_left, seed_right=seed_right, orthogonal=orthogonal, block_rks=block_rks, timer=timer)
        Ω .+= α[j] * Ωj
        Ψ .+= α[j] * Ψj
      end
    end

    result_rks = ones(Int, N+1)
    # Deep copy Ψ to avoid reference issues
    @timeit timer "rounding" begin
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
    end

    return TTvector{T,N}(N,Ψ_copy,dims,result_rks,zeros(N))
  end
end

function stta(α::Vector{T}, y::Vector{TTvector{T,N}}, rmax::Int; 
              seed_left::Int=1234, seed_right::Int=5678, orthogonal::Bool=true, block_rks::Int=N, timer::TimerOutput = TimerOutput()) where {T,N}
    # Generate uniform ranks with rmax
    rks = ones(Int, N+1)
    rks[2:N] .= rmax
    return stta(α, y, rks; orthogonal=orthogonal, seed_left=seed_left, seed_right=seed_right, block_rks=block_rks, timer=timer)
end


function my_qc!(A::Matrix{T}) where {T<:Number}
  m = size(A,1)
  n = size(A,2)
  if m>0 && n>0
    # Lapack in-place pivoted QR factorization
    A, tau, jpvt = LinearAlgebra.LAPACK.geqp3!(A)
    # Search for effective rank
    ϵ = 16 * A[1,1] * eps()
    rank = min(m,n) - searchsortedlast(view(A, reverse(diagind(A))), ϵ, by=abs)
    # Extract C = R*P' factor
    C = zeros(T, rank, n)
    for j=1:n, i=1:min(j,rank)
      C[i, jpvt[j]] = A[i,j]
    end
    # Extract Q factor into A
    LinearAlgebra.LAPACK.orgqr!(A, tau)
    Q = view(A,:,1:rank)
  else  # n = 0
    rank = 0
    Q = view(Matrix{T}(undef,m,rank),:,1:rank)
    C = Matrix{T}(undef,rank,n)
  end

  return Q, C, rank
end