"""
    tt_recursive_sketch([T=Float64,] [H::TToperator,] A::TTvector, rks;       orthogonal=true, reverse=true, seed=1234, block_rks=N, oversampling=1, timer) -> (W, sketch_rks)
    tt_recursive_sketch([T=Float64,] [H::TToperator,] A::TTvector, rmax::Int; orthogonal=true, reverse=true, seed=1234, block_rks=N, oversampling=1, timer) -> (W, sketch_rks)

Compute a recursive sketch of a TTvector A, or of the result of applying TToperator H to TTvector A,
without explicitly forming the product, using optimized random projections with adaptive normalization.

Generates sketch matrices W[k] for each TT core by recursively contracting with random
orthogonal (or normalized) tensors. The sketch provides a compressed representation
suitable for efficient randomized algorithms like ttrand_rounding and STTA.

# Arguments
- `T::Type{<:Number}`: Element type for random sketch tensors (optional, default: Float64)
- `H::TToperator{TH,N}`: TToperator to apply (optional)
- `A::TTvector{TA,N}`: Input TTvector to sketch
- `rks::Vector{Int}` or `rmax::Int`: Target sketch ranks (length N+1 with boundary conditions rks[1]=1 or rks[N+1]=1) or maximum sketch rank

# Keyword Arguments
- `orthogonal::Bool=true`: Generate orthogonal random tensors (via QR) for better numerical stability
- `reverse::Bool=true`: Sweep direction (true: right-to-left, false: left-to-right)
- `seed::Int=1234`: Random seed for reproducibility
- `block_rks::Int=N`: Block rank for sketching heuristic (controls sketch granularity)
- `oversampling=1`: overall multiplier for the number of blocks, useful for STTA oversampling in particular

# Returns
- `W::Vector{Matrix}`: Sketch matrices where W[k] has size (A.ttv_rks[k], sketch_rks[k])
- `sketch_rks::Vector{Int}`: Actual sketch ranks achieved (includes oversampling factor)

# Algorithm
Uses a **block sketching strategy** with adaptive normalization:

1. **Block Structure**: Creates p[k] sketch blocks per core, with p determined by
   oversampling heuristic: `p[k] = ceil(2*rks[k]/block_rks)`

2. **Normalization Strategy**:
   - **Orthogonal blocks**: QR with normalization `sqrt(dim*left_rank/right_rank)`
   - **Identity optimization**: When ranks permit, uses identity structure
   - **Random blocks**: Normalized by `1/sqrt(right_rank)`
   - **Final scaling**: Each W[k] scaled by `1/sqrt(p[k])` for block averaging

3. **Adaptive Rank Adjustment**: Block ranks may be updated for identity optimization,
   ensuring numerical efficiency while maintaining approximation quality.

The combination of QR orthogonalization and careful normalization provides excellent
numerical stability and consistent spectral properties across different rank regimes.

# References
- Randomized tensor train decomposition with block sketching
- See ttrand_rounding, stta for usage examples
"""
function tt_recursive_sketch(::Type{T}, A::TTvector{TA,N}, rks; orthogonal=true, reverse=true, seed=1234, block_rks::Int=N, p=0, oversampling=1, block_offset=nothing, timer::TimerOutput = TimerOutput()) where {T<:Number,TA<:Number,N}
  @timeit timer "tt_recursive_sketch" begin
    dims = A.ttv_dims
    TW = typeof(one(T)*one(TA))
    W = Vector{Array{TW,3}}(undef, N+1)

    if reverse
      @timeit timer "sketch_initialization" begin
        @assert rks[N+1] == 1 && A.ttv_rks[N+1] == 1
        block_rks_vec = ones(Int, N+1)
        block_rks_vec[1:N] .= block_rks
        for k=N:-1:1
          block_rks_vec[k] = min(block_rks_vec[k], dims[k]*block_rks_vec[k+1])
        end

        if p == 0
          p = compute_sketch_blocks_heuristic(rks, block_rks_vec, N; reverse=true)
        end
        if oversampling ≠ 1
          p[1:N] .= ceil.(Int, oversampling .* p[1:N])
        end

        W[N+1] = ones(TW,1,1,1)

        # Preallocate buffer for the entire loop
        max_sketch_buffer_size = maximum(block_rks_vec[k+1] * dims[k] * block_rks_vec[k] * p[k] for k in 1:N)
        sketch_buffer = Vector{T}(undef, max_sketch_buffer_size)

        # Batched-kernel buffer for the (a, z, β, p) AV intermediate.
        max_contract_buffer_size = 0
        for k in 1:N
          z = dims[k]
          a = A.ttv_rks[k]
          α = A.ttv_rks[k+1]
          b = block_rks_vec[k]
          β = block_rks_vec[k+1]
          buf, = contract_sketch_core_backwards_batched_buffers_size(z, a, α, β, b, p[k])
          max_contract_buffer_size = max(max_contract_buffer_size, buf)
        end
        contract_buffer = (Vector{TW}(undef, max_contract_buffer_size),)
      end

      # GC.@preserve sketch_buffer and contract_buffer over the entire loop —
      # both are unsafe_wrap'd inside generate_sketch_blocks and the contract
      # kernels, and without preserve Julia's escape analysis can free them
      # between calls while the wrapped Arrays are still alive. The resulting
      # heap corruption only manifests during a later GC sweep (typically deep
      # in the det baseline of a long-running sweep — a heisenbug).
      @timeit timer "contraction_reverse" begin
        GC.@preserve sketch_buffer contract_buffer begin
          @inbounds for k in N:-1:1
            @timeit timer "sketch_generation" begin
              # generate_sketch_blocks returns (β, z, b, p); the batched kernel
              # wants S in (z, β, b, p) layout so AV[a, z, β, p] reshapes
              # directly to (a, z·β, p) for the per-p inner gemm. This permute
              # is tiny (β·z·b·p ≈ 62 KB at the worst Matern bond) and
              # eliminates a 128 MB permute on AV inside the kernel.
              B_sketch_βzbp = generate_sketch_blocks(seed, k, (block_offset===nothing ? 0 : block_offset[k]), T, block_rks_vec[k+1], dims[k], block_rks_vec[k], p[k], orthogonal; reverse=true, buffer=sketch_buffer, timer=timer)
              B_sketch = permutedims(B_sketch_βzbp, (2, 1, 3, 4))  # (z, β, b, p)
            end

            @timeit timer "W_allocation" begin
              W[k] = zeros(TW, A.ttv_rks[k], block_rks_vec[k], p[k])
            end

            @timeit timer "tensor_contraction" begin
              if k < N
                V_batched = view(W[k+1], :, :, 1:p[k])
              else
                V_batched = repeat(W[k+1], 1, 1, p[k])  # W[N+1] = ones(1,1,1)
              end
              contract_sketch_core_backwards!(W[k], A.ttv_vec[k], B_sketch, V_batched; buffer=contract_buffer)
            end
          end
        end
      end
    else
      @timeit timer "sketch_initialization" begin
        @assert rks[1] == 1 && A.ttv_rks[1] == 1
        block_rks_vec = ones(Int, N+1)
        block_rks_vec[2:N+1] .= block_rks
        for k=1:N
          block_rks_vec[k+1] = min(block_rks_vec[k+1], dims[k]*block_rks_vec[k])
        end

        if p == 0
          p = compute_sketch_blocks_heuristic(rks, block_rks_vec, N; reverse=false)
        end

        if oversampling ≠ 1
          p[2:N+1] .= ceil.(Int, oversampling .* p[2:N+1])
        end
        W[1] = ones(TW,1,1,1)

        # Preallocate buffer for the entire loop
        max_sketch_buffer_size = maximum(block_rks_vec[k] * dims[k] * block_rks_vec[k+1] * p[k+1] for k in 1:N)
        sketch_buffer = Vector{T}(undef, max_sketch_buffer_size)

        max_contract_buffer_size = 0
        for k in 1:N
          z = dims[k]
          α = A.ttv_rks[k]
          a = A.ttv_rks[k+1]
          β = block_rks_vec[k]
          b = block_rks_vec[k+1]
          buf1, _ = contract_sketch_core_forwards_batched_buffers_size(z, α, a, β, b, p[k+1])
          max_contract_buffer_size = max(max_contract_buffer_size, buf1)
        end
        contract_buffer = (Vector{TW}(undef, max_contract_buffer_size), Vector{TW}(undef, max_contract_buffer_size))
      end

      # GC.@preserve: see backward branch for rationale.
      @timeit timer "contraction_forward" begin
        GC.@preserve sketch_buffer contract_buffer begin
          @inbounds for k in 1:N
            @timeit timer "sketch_generation" begin
              B_sketch = generate_sketch_blocks(seed, k, (block_offset===nothing ? 0 : block_offset[k]), T, block_rks_vec[k], dims[k], block_rks_vec[k+1], p[k+1], orthogonal; reverse=false, buffer=sketch_buffer, timer=timer)
            end

            @timeit timer "W_allocation" begin
              W[k+1] = zeros(TW, A.ttv_rks[k+1], block_rks_vec[k+1], p[k+1])
            end

            @timeit timer "tensor_contraction" begin
              if k > 1
                V_batched = view(W[k], :, :, 1:p[k+1])
              else
                V_batched = repeat(W[k], 1, 1, p[k+1])  # W[1] = ones(1,1,1) — tile.
              end
              contract_sketch_core_forwards!(W[k+1], A.ttv_vec[k], B_sketch, V_batched; buffer=contract_buffer)
            end
          end
        end
      end
    end

    for k=1:N+1
      W[k] ./= sqrt(p[k])
    end
    return [reshape(W[k], A.ttv_rks[k], block_rks_vec[k]*p[k]) for k=1:N+1], block_rks_vec.*p
  end
end

function tt_recursive_sketch(::Type{T}, A::TTvector{TA,N}, rmax::Int; orthogonal=true, reverse=true, seed=1234, block_rks::Int=N, p=0, timer::TimerOutput = TimerOutput()) where {T<:Number,TA<:Number,N}
  rks = rmax*ones(Int,N+1)
  rks[(reverse ? N+1 : 1)] = 1
  return tt_recursive_sketch(T,A,rks; orthogonal=orthogonal, reverse=reverse, seed=seed, block_rks=block_rks, p=p, timer=timer)
end

function tt_recursive_sketch(A::TTvector{T,N},rks_or_rmax; orthogonal=true, reverse=true, seed=1234, block_rks::Int=N, p=0, timer::TimerOutput = TimerOutput()) where {T<:Number,N}
  return tt_recursive_sketch(Float64,A,rks_or_rmax; orthogonal=orthogonal, reverse=reverse, seed=seed, block_rks=block_rks, p=p, timer=timer)
end

function tt_recursive_sketch(::Type{T}, H::TToperator{TH,N}, A::TTvector{TA,N}, rks; orthogonal=true, reverse=true, seed=1234, block_rks::Int=N, p=0, oversampling=1, block_offset=nothing, timer::TimerOutput = TimerOutput()) where {T<:Number,TA<:Number,TH<:Number,N}
  @timeit timer "tt_recursive_sketch" begin
    dims = A.ttv_dims
    TW = typeof(one(T)*one(TA)*one(TH))
    W = Vector{Array{TW,4}}(undef, N+1)

    if reverse
      @timeit timer "sketch_initialization" begin
        @assert rks[N+1] == 1 && A.ttv_rks[N+1] == 1 && H.tto_rks[N+1] == 1
        block_rks_vec = ones(Int, N+1)
        block_rks_vec[1:N] .= block_rks
        for k=N:-1:1
          block_rks_vec[k] = min(block_rks_vec[k], dims[k]*block_rks_vec[k+1])
        end

        if p == 0
          p = compute_sketch_blocks_heuristic(rks, block_rks_vec, N; reverse=true)
        end
        if oversampling ≠ 1
          p[1:N] .= ceil.(Int, oversampling .* p[1:N])
        end

        W[N+1] = ones(TW,1,1,1,1)

        # Preallocate buffer for the entire loop
        max_sketch_buffer_size = maximum(block_rks_vec[k+1] * dims[k] * block_rks_vec[k] * p[k] for k in 1:N)
        sketch_buffer = Vector{T}(undef, max_sketch_buffer_size)

        max_contract_buffer1_size = 0
        max_contract_buffer2_size = 0
        max_contract_buffer3_size = 0
        for k in 1:N
          z = dims[k]
          ζ = dims[k]
          a = A.ttv_rks[k]
          α = A.ttv_rks[k+1]
          b = H.tto_rks[k]
          β = H.tto_rks[k+1]
          c = block_rks_vec[k]
          γ = block_rks_vec[k+1]
          buf1_size, buf2_size, buf3_size = contract_sketch_core_backwards_operator_buffers_size(z, z, a, α, b, β, c, γ)
          max_contract_buffer1_size = max(max_contract_buffer1_size, buf1_size)
          max_contract_buffer2_size = max(max_contract_buffer2_size, buf2_size)
          max_contract_buffer3_size = max(max_contract_buffer3_size, buf3_size)
        end
        contract_buffer = (Vector{TW}(undef, max_contract_buffer1_size),
                           Vector{TW}(undef, max_contract_buffer2_size),
                           Vector{TW}(undef, max_contract_buffer3_size))
      end

      @timeit timer "contraction_reverse" begin
        @inbounds for k in N:-1:1
          @timeit timer "sketch_generation" begin
            B_sketch = generate_sketch_blocks(seed, k, (block_offset===nothing ? 0 : block_offset[k]), T, block_rks_vec[k+1], dims[k], block_rks_vec[k], p[k], orthogonal; reverse=true, timer=timer, buffer=sketch_buffer)
          end

          @timeit timer "W_allocation" begin
            W[k] = zeros(TW, A.ttv_rks[k], H.tto_rks[k], block_rks_vec[k], p[k])
          end

          @timeit timer "tensor_contraction" begin
            for j=1:p[k]
              W_next_j = view(W[k+1],:,:,:,(k<N ? j : 1))
              B_j = view(B_sketch,:,:,:,j)
              W_k_j = view(W[k],:,:,:,j)
              contract_sketch_core_backwards!(W_k_j, A.ttv_vec[k], H.tto_vec[k], B_j, W_next_j; buffer=contract_buffer)
            end
          end
        end
      end
    else
      @timeit timer "sketch_initialization" begin
        @assert rks[1] == 1 && A.ttv_rks[1] == 1 && H.tto_rks[1] == 1
        block_rks_vec = ones(Int, N+1)
        block_rks_vec[2:N+1] .= block_rks
        for k=1:N
          block_rks_vec[k+1] = min(block_rks_vec[k+1], dims[k]*block_rks_vec[k])
        end

        if p == 0
          p = compute_sketch_blocks_heuristic(rks, block_rks_vec, N; reverse=false)
        end
        # No oversampling?

        W[1] = ones(TW,1,1,1,1)

        # Preallocate buffer for the entire loop
        max_sketch_buffer_size = maximum(block_rks_vec[k] * dims[k] * block_rks_vec[k+1] * p[k] for k in 1:N)
        sketch_buffer = Vector{T}(undef, max_sketch_buffer_size)

        max_contract_buffer1_size = 0
        max_contract_buffer2_size = 0
        max_contract_buffer3_size = 0
        for k in 1:N
          ζ = dims[k]
          z = dims[k]
          α = A.ttv_rks[k]
          a = A.ttv_rks[k+1]
          β = H.tto_rks[k]
          b = H.tto_rks[k+1]
          γ = block_rks_vec[k]
          c = block_rks_vec[k+1]
          buf1_size, buf2_size, buf3_size = contract_sketch_core_forwards_operator_buffers_size(z, z, α, a, β, b, γ, c)
          max_contract_buffer1_size = max(max_contract_buffer1_size, buf1_size)
          max_contract_buffer2_size = max(max_contract_buffer2_size, buf2_size)
          max_contract_buffer3_size = max(max_contract_buffer3_size, buf3_size)
        end
        contract_buffer = (Vector{TW}(undef, max_contract_buffer1_size),
                           Vector{TW}(undef, max_contract_buffer2_size),
                           Vector{TW}(undef, max_contract_buffer3_size))
      end

      @timeit timer "contraction_forward" begin
        @inbounds for k in 1:N
          @timeit timer "sketch_generation" begin
            B_sketch = generate_sketch_blocks(seed, k, (block_offset===nothing ? 0 : block_offset[k]), T, block_rks_vec[k], dims[k], block_rks_vec[k+1], p[k+1], orthogonal; reverse=false, timer=timer)
          end

          @timeit timer "W_allocation" begin
            W[k+1] = zeros(TW, A.ttv_rks[k+1], H.tto_rks[k+1], block_rks_vec[k+1], p[k+1])
          end

          @timeit timer "tensor_contraction" begin
            for j=1:p[k+1]
              W_k_j = view(W[k],:,:,:,(k>1 ? j : 1))
              B_j = view(B_sketch,:,:,:,j)
              W_next_j = view(W[k+1],:,:,:,j)
              contract_sketch_core_forwards!(W_next_j, A.ttv_vec[k], H.tto_vec[k], B_j, W_k_j; buffer=contract_buffer)

              # @tensoropt((a,b,h,α,β,η), W_next_j[a,h,b] = A.ttv_vec[k][z,α,a]*H.tto_vec[k][y,z,η,h]*B_j[y,β,b]*W_k_j[α,η,β])
            end
          end
        end
      end
    end

    for k=1:N+1
      W[k] ./= sqrt(p[k])
    end
    return [reshape(W[k], A.ttv_rks[k], H.tto_rks[k], block_rks_vec[k]*p[k]) for k=1:N+1], block_rks_vec.*p
  end
end

function tt_recursive_sketch(::Type{T}, H::TToperator{TH,N}, A::TTvector{TA,N}, rmax::Int; orthogonal=true, reverse=true, seed=1234, block_rks::Int=N, p=0, timer::TimerOutput = TimerOutput()) where {T<:Number,TA<:Number,TH<:Number,N}
  d = A.N
  rks = rmax*ones(Int,d+1)
  rks[(reverse ? d+1 : 1)] = 1
  return tt_recursive_sketch(T,H,A,rks; orthogonal=orthogonal, reverse=reverse, seed=seed, block_rks=block_rks, p=p, timer=timer)
end

function tt_recursive_sketch(H::TToperator{TH,N}, A::TTvector{TA,N}, rks_or_rmax; orthogonal=true, reverse=true, seed=1234, block_rks::Int=N, p=0, timer::TimerOutput = TimerOutput()) where {TA<:Number,TH<:Number,N}
  return tt_recursive_sketch(Float64, H, A, rks_or_rmax; orthogonal=orthogonal, reverse=reverse, seed=seed, block_rks=block_rks, p=p, timer=timer)
end

function tt_recursive_sketch(::Type{T}, A::NTuple{M,TTvector{TA,N}}, rks; orthogonal=true, reverse=true, seed=1234, block_rks::Int=N, p=0, oversampling=1, block_offset=nothing, timer::TimerOutput = TimerOutput()) where {T<:Number,TA<:Number,N,M}
  @timeit timer "tt_recursive_sketch" begin
    dims = A[1].ttv_dims
    @assert all(a.ttv_dims == dims for a in A)
    TW = typeof(one(T)*one(TA))
    W = Vector{Array{TW,M+2}}(undef, N+1)

    if reverse
      @timeit timer "sketch_initialization" begin
        @assert rks[N+1] == 1 && all(a.ttv_rks[N+1] == 1 for a in A)
        block_rks_vec = ones(Int, N+1)
        block_rks_vec[1:N] .= block_rks
        for k=N:-1:1
          block_rks_vec[k] = min(block_rks_vec[k], dims[k]*block_rks_vec[k+1])
        end

        if p == 0
          p = compute_sketch_blocks_heuristic(rks, block_rks_vec, N; reverse=true)
        end
        if oversampling ≠ 1
          p[1:N] .= ceil.(Int, oversampling .* p[1:N])
        end

        W[N+1] = ones(TW, ntuple(i->1, M+2))

        # Preallocate buffer for the entire loop
        max_sketch_buffer_size = maximum(block_rks_vec[k+1] * dims[k] * block_rks_vec[k] * p[k] for k in 1:N)
        sketch_buffer = Vector{T}(undef, max_sketch_buffer_size)

        max_contract_buffer1_size = 0
        max_contract_buffer2_size = 0
        max_contract_buffer3_size = 0
        for k in 1:N
          z = dims[k]
          v = ntuple(i->(i==M+1 ? block_rks_vec[k+1] : A[i].ttv_rks[k+1]), M+1)
          w = ntuple(i->(i==M+1 ? block_rks_vec[k  ] : A[i].ttv_rks[k  ]), M+1)
          buf1_size, buf2_size, buf3_size = contract_sketch_core_backwards_kronecker_buffers_size(z, v, w)
          max_contract_buffer1_size = max(max_contract_buffer1_size, buf1_size)
          max_contract_buffer2_size = max(max_contract_buffer2_size, buf2_size)
          max_contract_buffer3_size = max(max_contract_buffer3_size, buf3_size)
        end
        contract_buffer = (Vector{TW}(undef, max_contract_buffer1_size), Vector{TW}(undef, max_contract_buffer2_size), Vector{TW}(undef, max_contract_buffer3_size))
      end

      @timeit timer "contraction_reverse" begin
        @inbounds for k in N:-1:1
          @timeit timer "sketch_generation" begin
            B_sketch = generate_sketch_blocks(seed, k, (block_offset===nothing ? 0 : block_offset[k]), T, block_rks_vec[k+1], dims[k], block_rks_vec[k], p[k], orthogonal; reverse=true, buffer=sketch_buffer, timer=timer)
          end

          @timeit timer "W_allocation" begin
            W[k] = zeros(TW, (a.ttv_rks[k] for a in A)..., block_rks_vec[k], p[k])
          end

          @timeit timer "tensor_contraction" begin
            for j=1:p[k]
              W_next_j = view(W[k+1], ntuple(i->Colon(), M+1)..., (k<N ? j : 1))
              B_j = view(B_sketch,:,:,:,j)
              W_k_j = view(W[k], ntuple(i->Colon(), M+1)..., j)
              contract_sketch_core_kronecker_backwards!(W_k_j, ntuple(i -> A[i].ttv_vec[k], M), B_j, W_next_j; buffer=contract_buffer)
            end
          end
        end
      end
    else
      @timeit timer "sketch_initialization" begin
        @assert rks[1] == 1 && all(a.ttv_rks[1] == 1 for a in A)
        block_rks_vec = ones(Int, N+1)
        block_rks_vec[2:N+1] .= block_rks
        for k=1:N
          block_rks_vec[k+1] = min(block_rks_vec[k+1], dims[k]*block_rks_vec[k])
        end

        if p == 0
          p = compute_sketch_blocks_heuristic(rks, block_rks_vec, N; reverse=false)
        end
        if oversampling ≠ 1
          p[2:N+1] .= ceil.(Int, oversampling .* p[2:N+1])
        end
        W[1] = ones(TW, ntuple(i->1, M+2))

        # Preallocate buffer for the entire loop
        max_sketch_buffer_size = maximum(block_rks_vec[k] * dims[k] * block_rks_vec[k+1] * p[k+1] for k in 1:N)
        sketch_buffer = Vector{T}(undef, max_sketch_buffer_size)

        max_contract_buffer1_size = 0
        max_contract_buffer2_size = 0
        max_contract_buffer3_size = 0
        for k in 1:N
          z = dims[k]
          v = ntuple(i->(i==M+1 ? block_rks_vec[k  ] : A[i].ttv_rks[k  ]), M+1)
          w = ntuple(i->(i==M+1 ? block_rks_vec[k+1] : A[i].ttv_rks[k+1]), M+1)
          buf1_size, buf2_size, buf3_size = contract_sketch_core_forwards_kronecker_buffers_size(z, v, w)
          max_contract_buffer1_size = max(max_contract_buffer1_size, buf1_size)
          max_contract_buffer2_size = max(max_contract_buffer2_size, buf2_size)
          max_contract_buffer3_size = max(max_contract_buffer3_size, buf3_size)
        end
        contract_buffer = (Vector{TW}(undef, max_contract_buffer1_size), Vector{TW}(undef, max_contract_buffer2_size), Vector{TW}(undef, max_contract_buffer3_size))
      end

      @timeit timer "contraction_forward" begin
        @inbounds for k in 1:N
          @timeit timer "sketch_generation" begin
            B_sketch = generate_sketch_blocks(seed, k, (block_offset===nothing ? 0 : block_offset[k]), T, block_rks_vec[k], dims[k], block_rks_vec[k+1], p[k+1], orthogonal; reverse=false, buffer=sketch_buffer, timer=timer)
          end

          @timeit timer "W_allocation" begin
            W[k+1] = zeros(TW, (a.ttv_rks[k+1] for a in A)..., block_rks_vec[k+1], p[k+1])
          end

          @timeit timer "tensor_contraction" begin
            for j=1:p[k+1]
              W_k_j = view(W[k], ntuple(i->( i<M+2 ? Colon() : (k>1 ? j : 1) ), M+2)...)
              B_j = view(B_sketch,:,:,:,j)
              W_next_j = view(W[k+1], ntuple(i->( i<M+2 ? Colon() : j ), M+2)...)
              contract_sketch_core_kronecker_forwards!(W_next_j, ntuple(i -> A[i].ttv_vec[k], M), B_j, W_k_j; buffer=contract_buffer)
            end
          end
        end
      end
    end

    for k=1:N+1
      W[k] ./= sqrt(p[k])
    end
    return [reshape(W[k], (a.ttv_rks[k] for a in A)..., block_rks_vec[k]*p[k]) for k=1:N+1], block_rks_vec.*p
  end
end

function tt_recursive_sketch(::Type{T}, A::NTuple{M,TTvector{TA,N}}, rmax::Int; orthogonal=true, reverse=true, seed=1234, block_rks::Int=N, p=0, timer::TimerOutput = TimerOutput()) where {T<:Number,TA<:Number,N,M}
  rks = rmax*ones(Int,N+1)
  rks[(reverse ? N+1 : 1)] = 1
  return tt_recursive_sketch(T,A,rks; orthogonal=orthogonal, reverse=reverse, seed=seed, block_rks=block_rks, p=p, timer=timer)
end

function tt_recursive_sketch(A::NTuple{M,TTvector{T,N}},rks_or_rmax; orthogonal=true, reverse=true, seed=1234, block_rks::Int=N, p=0, timer::TimerOutput = TimerOutput()) where {T<:Number,N,M}
  return tt_recursive_sketch(Float64,A,rks_or_rmax; orthogonal=orthogonal, reverse=reverse, seed=seed, block_rks=block_rks, p=p, timer=timer)
end
