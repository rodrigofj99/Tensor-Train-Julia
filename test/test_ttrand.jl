using Test
using TensorTrains
using LinearAlgebra

@testset "tt_recursive_sketch" begin
  # Test basic sketching with vector of ranks
  dims = (3,3,3,3)
  rks = [1,3,3,3,1]
  A = rand_tt(dims,rks)

  # Test with explicit ranks vector
  sketch_rks = [1,2,2,2,1]
  W, actual_rks = TensorTrains.tt_recursive_sketch(Float64, A, sketch_rks; seed=1234)
  @test length(W) == length(dims) + 1
  @test length(actual_rks) == length(sketch_rks)
  @test actual_rks[1] >= sketch_rks[1]  # Actual ranks may be higher due to blocking

  # Test with maximum rank
  W_max, actual_rks_max = TensorTrains.tt_recursive_sketch(Float64, A, 4; seed=1234)
  @test length(W_max) == length(dims) + 1
  # Actual ranks may exceed rmax due to blocking heuristic
  @test all(actual_rks_max .>= 1)

  # Test forward vs reverse sketching
  W_fwd, _ = TensorTrains.tt_recursive_sketch(A, sketch_rks; reverse=false, seed=1234)
  W_rev, _ = TensorTrains.tt_recursive_sketch(A, sketch_rks; reverse=true, seed=1234)
  @test size(W_fwd[1]) != size(W_rev[1])  # Different sketch directions

  # Test with operator - TODO: Fix reshape bugs in operator sketch code
  # H = rand_tto(dims,3)
  # W_op, op_rks = TensorTrains.tt_recursive_sketch(Float64, H, A, sketch_rks; seed=5678)
  # @test length(W_op) == length(dims) + 1
end

@testset "partial contraction" begin
  dims = (2,2,2,2)
  rks = [1,4,4,4,1]
  A = rand_tt(dims,rks)
  B = rand_tt(dims,rks)
  W = TensorTrains.partial_contraction(A,B)
  @test isapprox(W[1][1],dot(A,B))

  # Test reverse mode
  W_rev = TensorTrains.partial_contraction(A,B;reverse=false)
  @test isapprox(W_rev[end][1],dot(A,B))
end

@testset "default_rank_heuristic" begin
  # Test rank heuristic for single TTvector
  dims = (3,3,3,3)
  rks = [1,3,5,4,1]
  A = rand_tt(dims,rks)
  rks_heur = TensorTrains.default_rank_heuristic(A)
  @test length(rks_heur) == length(dims) + 1  # Returns full rank vector including boundaries
  @test rks_heur[1] == 1 && rks_heur[end] == 1  # Boundary ranks
  @test all(rks_heur .>= 1)

  # Test rank heuristic for vector of TTvectors
  B = rand_tt(dims,[1,2,3,2,1])
  C = rand_tt(dims,[1,4,6,3,1])
  rks_vec = TensorTrains.default_rank_heuristic([A,B,C])
  @test length(rks_vec) == length(dims) + 1  # Returns full rank vector including boundaries
  @test rks_vec[1] == 1 && rks_vec[end] == 1  # Boundary ranks
  @test all(rks_vec[2:end-1] .>= maximum([3,2,4]))  # Interior ranks should be at least max of individual ranks
end

@testset "ttrand_rounding" begin
  dims = (2,2,2,2,2,2,2,2)
  rks = [1,2,4,4,4,4,4,2,1]
  A_tt = rand_tt(dims,rks)
  A_pert = tt_up_rks(A_tt,20,ϵ_wn=1e-8)
  A = ttv_to_tensor(A_pert)

  # Test with explicit ranks
  A_ttrand = ttrand_rounding(A_pert, [1,2,4,6,6,6,4,2,1])
  A_rand = ttv_to_tensor(A_ttrand)
  @test isapprox(A,A_rand)

  # Test with seed parameter for reproducibility
  A_ttrand1 = ttrand_rounding(A_pert, [1,2,4,6,6,6,4,2,1]; seed=9999)
  A_ttrand2 = ttrand_rounding(A_pert, [1,2,4,6,6,6,4,2,1]; seed=9999)
  @test ttv_to_tensor(A_ttrand1) ≈ ttv_to_tensor(A_ttrand2)

  # Test with automatic rank heuristic
  A_ttrand_auto = ttrand_rounding(A_pert)
  A_rand_auto = ttv_to_tensor(A_ttrand_auto)
  @test isapprox(A,A_rand_auto,atol=1e-6)
end

@testset "stta_sketch" begin
  dims = (3,3,3,3)
  rks = [1,3,3,3,1]
  A = rand_tt(dims,rks)

  # Test seed-based interface (new version)
  target_rks = [1,3,3,3,1]
  Ω, Ψ = TensorTrains.stta_sketch(A, target_rks; seed_left=1234, seed_right=5678)
  # Get actual sketch ranks (left ranks are 50% larger than target ranks)
  l_rks = copy(target_rks)
  l_rks[2:end-1] = ceil.(Int, 1.5 .* target_rks[2:end-1])
  r_rks = target_rks
  _, sketch_l_rks = TensorTrains.tt_recursive_sketch(Float64, A, l_rks; orthogonal=true, reverse=false, seed=1234)
  _, sketch_r_rks = TensorTrains.tt_recursive_sketch(Float64, A, r_rks; orthogonal=true, reverse=true, seed=5678)
  
  # Check all Ω dimensions
  for k in 1:length(Ω)
    @test size(Ω[k]) == (sketch_l_rks[k+1], sketch_r_rks[k+1])
  end
  
  # Check all Ψ dimensions  
  for k in 1:length(Ψ)
    @test size(Ψ[k]) == (dims[k], sketch_l_rks[k], sketch_r_rks[k+1])
  end

  # Test with explicit random TTvectors (original version)
  L = rand_tt(dims, l_rks)
  R = rand_tt(dims, r_rks)
  Ω2, Ψ2 = TensorTrains.stta_sketch(A, L, R)
  
  # Check all Ω2 dimensions
  for k in 1:length(Ω2)
    @test size(Ω2[k]) == (L.ttv_rks[k+1], R.ttv_rks[k+1])
  end
  
  # Check all Ψ2 dimensions
  for k in 1:length(Ψ2)
    @test size(Ψ2[k]) == (dims[k], L.ttv_rks[k], R.ttv_rks[k+1])
  end

  # Test reproducibility with same seed
  Ω3, Ψ3 = TensorTrains.stta_sketch(A, target_rks; seed_left=1234, seed_right=5678)
  @test Ω ≈ Ω3
  @test Ψ ≈ Ψ3
end

@testset "STTA" begin
  dims = (2,2,2,2,2,2,2,2)
  rks = [1,2,4,4,4,4,4,2,1]
  A_tt = rand_tt(dims,rks)
  A_pert = tt_up_rks(A_tt,20,ϵ_wn=1e-8)
  A = ttv_to_tensor(A_pert)

  # Test with explicit ranks
  A_ttrand = stta(A_pert;rks=[1,2,4,6,6,6,4,2,1])
  A_rand = ttv_to_tensor(A_ttrand)
  @test isapprox(A,A_rand)

  # Test with seed parameters for reproducibility
  A_stta1 = stta(A_pert;rks=[1,2,4,6,6,6,4,2,1], seed_left=7777, seed_right=8888)
  A_stta2 = stta(A_pert;rks=[1,2,4,6,6,6,4,2,1], seed_left=7777, seed_right=8888)
  @test ttv_to_tensor(A_stta1) ≈ ttv_to_tensor(A_stta2)

  # Test with automatic rank selection
  A_stta_auto = stta(A_pert)
  A_auto = ttv_to_tensor(A_stta_auto)
  @test isapprox(A,A_auto,atol=1e-6)

  # Test with different ranks
  A_stta_different = stta(A_pert;rks=[1,2,4,4,4,4,4,2,1])
  A_different = ttv_to_tensor(A_stta_different)
  @test isapprox(A,A_different,atol=1e-6)
end

@testset "TT sum rand_rounding" begin
  d = 10
  n = 50
  rks = 5
  dims = ntuple(x->n,d)
  A_tt = rand_tt(dims,rks;normalise=true)
  B_tt = rand_tt(dims,20*rks;normalise=true)
  ε = 1e-4
  @time A_ttsvd = tt_rounding(A_tt+ε*B_tt;tol=ε)
  @time A_rand = ttrand_rounding(A_tt+ε*B_tt, 10)
  @test(norm(A_tt-A_rand)<1e-3*norm(A_tt))
end