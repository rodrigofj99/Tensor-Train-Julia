using Test
using TensorTrains
using TensorTrains: cached_operator_sketch, cached_sketch, cached_kronecker_sketch
using LinearAlgebra
using Random

BLAS.set_num_threads(1)   # determinism for the cache on/off bit-identity checks

@testset "ttrand_rounding_adaptive — low-rank sanity" begin
  Random.seed!(42)
  dims = (4, 4, 4, 4, 4)
  rks_true = [1, 3, 4, 4, 3, 1]
  y_lr = rand_tt(Float64, dims, rks_true; orthogonal=true)
  y_inflated = tt_up_rks(y_lr, 12; ϵ_wn=1e-12)

  ŷ = ttrand_rounding_adaptive(y_inflated, 1e-8; seed=99)
  err = norm(y_inflated - ŷ) / norm(y_inflated)
  @test err < 1e-7
  # Output rank at each bond should not exceed the true low rank (up to small slack)
  @test all(ŷ.ttv_rks[k] <= rks_true[k] + 2 for k = 1:length(dims)+1)
end

@testset "ttrand_rounding_adaptive — decaying spectrum" begin
  Random.seed!(2024)
  N = 8
  dims = ntuple(i -> 2, N)
  psi = rand_tt(Float64, dims, 16; orthogonal=true)
  v = tt_to_vidal(psi)
  decay_rate = 0.5
  for i = 1:length(v.Σ)
    v.Σ[i] .*= exp.(-decay_rate .* (axes(v.Σ[i], 1) .- 1))
  end
  psi = vidal_to_left_canonical(v)
  psi = psi / norm(psi)
  y_norm = norm(psi)

  prev_max_rk = 0  # sentinel: first iteration's monotonicity check is trivially satisfied
  # Randomized methods have a noise floor — accept achieved error vs the larger of
  # 10ε and the floor.
  noise_floor = 1e-6
  for ε in (1e-2, 1e-4, 1e-6, 1e-8)
    ŷ = ttrand_rounding_adaptive(psi, ε; seed=12345)
    err = norm(psi - ŷ) / y_norm
    @test err <= max(10 * ε, noise_floor)

    # Monotonicity: smaller ε should not yield smaller ranks
    max_rk = maximum(ŷ.ttv_rks)
    @test max_rk >= prev_max_rk - 1  # allow ±1 fluctuation from randomness
    prev_max_rk = max_rk
  end
end

@testset "ttrand_rounding_adaptive — N=1 edge case" begin
  y = rand_tt(Float64, (5,), [1, 1])
  ŷ = ttrand_rounding_adaptive(y, 1e-6)
  @test ŷ.ttv_rks == y.ttv_rks
  @test size(ŷ.ttv_vec[1]) == size(y.ttv_vec[1])
  @test ŷ.ttv_vec[1] ≈ y.ttv_vec[1]
end

@testset "ttrand_rounding_adaptive — linear combination" begin
  Random.seed!(11)
  N = 5
  dims = ntuple(i -> 3, N)
  rks = [1, 3, 5, 5, 3, 1]
  y_arr = [rand_tt(Float64, dims, rks; orthogonal=true) for _ = 1:3]
  α = [0.7, -0.4, 1.1]
  ref = α[1]*y_arr[1] + α[2]*y_arr[2] + α[3]*y_arr[3]
  ref_norm = norm(ref)
  for ε in (1e-2, 1e-6, 1e-10)
    ŷ = ttrand_rounding_adaptive(α, y_arr, ε; seed=2024)
    err = norm(ref - ŷ) / ref_norm
    @test err <= max(10 * ε, 1e-12)
  end
end

@testset "ttrand_rounding_adaptive — operator times vector minus vector" begin
  Random.seed!(12)
  N = 5
  dims = ntuple(i -> 3, N)
  A = rand_tto(dims, 2)
  y = rand_tt(Float64, dims, [1,2,3,3,2,1]; orthogonal=true)
  b = rand_tt(Float64, dims, [1,2,2,2,2,1]; orthogonal=true)
  ref = A*y - b
  ref_norm = norm(ref)
  for ε in (1e-2, 1e-6, 1e-10)
    ŷ = ttrand_rounding_adaptive(A, y, b, ε; seed=2024)
    err = norm(ref - ŷ) / ref_norm
    @test err <= max(10 * ε, 1e-12)
  end

  # Reusable caches (operator sketch of A·y + vector sketch of b) reproduce the throwaway-cache
  # result and survive reuse after growth (single-thread BLAS).
  ε = 1e-4; brk = N; sd = 2024; nsamp = max(N÷2, 4); init = 4 + nsamp
  r0  = ttrand_rounding_adaptive(A, y, b, ε; seed=sd, block_rks=brk)
  opc = cached_operator_sketch(Float64, A, y, brk, brk, init; seed=sd)
  bc  = cached_sketch(Float64, b, brk, brk, init; seed=sd)
  r1  = ttrand_rounding_adaptive(A, y, b, ε; seed=sd, block_rks=brk, caches=(opc, bc))
  r2  = ttrand_rounding_adaptive(A, y, b, ε; seed=sd, block_rks=brk, caches=(opc, bc))  # reuse grown
  @test norm(r1 - r0) / norm(r0) < 1e-12
  @test norm(r2 - r0) / norm(r0) < 1e-12
end

@testset "ttrand_rounding_adaptive — mixed operator combination" begin
  Random.seed!(21)
  N = 5
  dims = ntuple(i -> 3, N)
  A  = rand_tto(dims, 2)
  y1 = rand_tt(Float64, dims, [1,2,3,3,2,1]; orthogonal=true)
  y2 = rand_tt(Float64, dims, [1,2,2,2,2,1]; orthogonal=true)
  y3 = rand_tt(Float64, dims, [1,2,2,3,2,1]; orthogonal=true)
  α = [1.5, -2.0, 0.7]; y = [y1, y2, y3]
  ref = α[1]*(A*y1) + α[2]*y2 + α[3]*y3
  ref_norm = norm(ref)
  for ε in (1e-2, 1e-6, 1e-10)
    ŷ = ttrand_rounding_adaptive(α, A, y, ε; seed=2024)
    @test norm(ref - ŷ) / ref_norm <= max(10 * ε, 1e-12)
  end

  # Reusable caches (op cache for A·y[1] + vector caches for y[2..]) reproduce the throwaway result.
  brk = N; sd = 2024; init = 4 + max(N÷2, 4)
  r0  = ttrand_rounding_adaptive(α, A, y, 1e-4; seed=sd, block_rks=brk)
  ch  = Any[cached_operator_sketch(Float64, A, y1, brk, brk, init; seed=sd),
            cached_sketch(Float64, y2, brk, brk, init; seed=sd),
            cached_sketch(Float64, y3, brk, brk, init; seed=sd)]
  r1  = ttrand_rounding_adaptive(α, A, y, 1e-4; seed=sd, block_rks=brk, caches=ch)
  r2  = ttrand_rounding_adaptive(α, A, y, 1e-4; seed=sd, block_rks=brk, caches=ch)  # reuse grown
  @test norm(r1 - r0) / norm(r0) < 1e-12
  @test norm(r2 - r0) / norm(r0) < 1e-12
end

@testset "ttrand_rounding_adaptive — Hadamard product" begin
  Random.seed!(13)
  N = 5
  dims = ntuple(i -> 3, N)
  ya = rand_tt(Float64, dims, [1,2,3,3,2,1]; orthogonal=true)
  yb = rand_tt(Float64, dims, [1,2,3,3,2,1]; orthogonal=true)
  ref_dense = ttv_to_tensor(ya) .* ttv_to_tensor(yb)
  ref_norm = norm(ref_dense)
  for ε in (1e-2, 1e-6, 1e-10)
    ŷ = ttrand_rounding_adaptive((ya, yb), ε; seed=2024)
    err = norm(ttv_to_tensor(ŷ) - ref_dense) / ref_norm
    @test err <= max(10 * ε, 1e-12)
  end

  # M = 1 should fall through to the single-TT method
  ŷ1 = ttrand_rounding_adaptive((ya,), 1e-8; seed=2024)
  err1 = norm(ya - ŷ1) / norm(ya)
  @test err1 <= 1e-7

  # Reusable two-group Kronecker cache reproduces the throwaway-cache result and survives reuse.
  brk = N; brk_inc = max(1, N÷4); sd = 2024
  ℓmax = prod(maximum(ya.ttv_rks) for _ in 1:1) * maximum(yb.ttv_rks)
  nsamp = max(N÷2, 4, ceil(Int, 0.1*ℓmax))
  r0 = ttrand_rounding_adaptive((ya, yb), 1e-4; seed=sd)
  c  = cached_kronecker_sketch(Float64, (ya, yb), brk, brk_inc, 4+nsamp; seed=sd)
  r1 = ttrand_rounding_adaptive((ya, yb), 1e-4; seed=sd, cache=c)
  r2 = ttrand_rounding_adaptive((ya, yb), 1e-4; seed=sd, cache=c)   # reuse grown
  @test norm(r1 - r0) / norm(r0) < 1e-12
  @test norm(r2 - r0) / norm(r0) < 1e-12
end

@testset "ttrand_rounding_adaptive — ε monotonicity" begin
  # The estimator must actually detect residuals: tighter ε should produce
  # equal-or-larger ranks on a TT with non-trivial structure.
  Random.seed!(7)
  dims = (3, 3, 3, 3, 3)
  y = rand_tt(Float64, dims, [1, 4, 6, 4, 3, 1]; orthogonal=true)
  y_inflated = tt_up_rks(y, 10; ϵ_wn=1e-2)
  ŷ_loose = ttrand_rounding_adaptive(y_inflated, 1.0;  seed=1)
  ŷ_tight = ttrand_rounding_adaptive(y_inflated, 1e-8; seed=1)
  @test maximum(ŷ_tight.ttv_rks) >= maximum(ŷ_loose.ttv_rks)
end
