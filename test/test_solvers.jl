using TensorTrains, Test, LinearAlgebra, Random

BLAS.set_num_threads(1)   # determinism for the reuse on/off bit-identity checks

@testset "sketched_gmres — window-sketch reuse on/off bit-identical" begin
    Random.seed!(5)
    N = 5; n = 2; dims = ntuple(i -> n, N)
    # Well-conditioned operator I + small perturbation so sketched GMRES converges quickly.
    A = id_tto(Float64, N; n_dim=n) + 0.03 * rand_tto(dims, 2)
    xtrue = rand_tt(Float64, dims, 3; orthogonal=true)
    b  = A * xtrue
    x0 = 0.0 * xtrue

    @testset "TToperator path" begin
        xT, hT = sketched_gmres(A, b, x0; m=20, tol=1e-6, rmax=64, block_rks=4, verbose=false, reuse_sketches=true)
        xF, hF = sketched_gmres(A, b, x0; m=20, tol=1e-6, rmax=64, block_rks=4, verbose=false, reuse_sketches=false)
        @test length(hT) == length(hF)                       # same iteration count
        @test norm(xT - xF) / norm(xF) < 1e-12               # reuse on == off (bit-identical)
        @test norm(A * xT - b) / norm(b) < 1e-3              # actually converges
    end

    @testset "function (summand) path" begin
        opfun = x -> [A * x]                                  # single-summand sum-operator form
        xT, hT = sketched_gmres(opfun, b, x0; m=20, tol=1e-6, rmax=64, block_rks=4, verbose=false, reuse_sketches=true)
        xF, hF = sketched_gmres(opfun, b, x0; m=20, tol=1e-6, rmax=64, block_rks=4, verbose=false, reuse_sketches=false)
        @test length(hT) == length(hF)
        @test norm(xT - xF) / norm(xF) < 1e-12
        @test norm(A * xT - b) / norm(b) < 1e-3
    end
end
