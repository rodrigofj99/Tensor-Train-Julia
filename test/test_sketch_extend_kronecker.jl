using TensorTrains, Test, LinearAlgebra
using TensorTrains: KroneckerSketchGroup, extend_kronecker_sketch!, finalize_cols_kronecker,
                    tt_recursive_sketch, KroneckerCachedSketch, cached_kronecker_sketch,
                    ensure_columns!, sketch_matrix

BLAS.set_num_threads(1)   # determinism for the bit-identity checks

# Build a single Kronecker group up to per-bond counts `p` (length N+1, non-decreasing over 1:N).
function build_kron_group(A, block_rks, group, p; seed, orthogonal)
    g = KroneckerSketchGroup(Float64, A, block_rks, group)
    extend_kronecker_sketch!(g, A, p; seed=seed, orthogonal=orthogonal)
    return g
end

@testset "extend_kronecker_sketch! / finalize_cols_kronecker" begin
    dims = (4, 3, 5, 3, 4); N = 5; seed = 717
    ya = rand_tt(Float64, dims, 4; seed=11)
    yb = rand_tt(Float64, dims, 3; seed=22)
    yc = rand_tt(Float64, dims, 2; seed=33)
    prks2(l) = ya.ttv_rks[l]*yb.ttv_rks[l]
    prks3(l) = ya.ttv_rks[l]*yb.ttv_rks[l]*yc.ttv_rks[l]

    @testset "single group == tt_recursive_sketch(tuple) (bit-identical)" begin
        for (A, prks) in (((ya, yb), prks2), ((ya, yb, yc), prks3)), orth in (true, false), b in (3, 6)
            Wref, skref = tt_recursive_sketch(Float64, A, 12; seed=seed, block_rks=b, orthogonal=orth)
            brv = ones(Int, N+1); brv[1:N] .= b
            for k=N:-1:1; brv[k] = min(brv[k], dims[k]*brv[k+1]); end
            p = skref .÷ brv
            g = build_kron_group(A, b, 0, p; seed=seed, orthogonal=orth)
            for l in 1:N+1
                fin = finalize_cols_kronecker([g], l, prks(l); weighting=:equal)
                M = length(A)
                ref2d = reshape(Wref[l], prks(l), :)   # reference is (rks₁…rks_M, cols)
                @test fin == ref2d                     # bit-identical (group 0 ⇒ same blocks)
            end
        end
    end

    @testset "one-shot vs two-shot extension (bit-identical, history-independent)" begin
        A = (ya, yb); b = 4
        Wref, skref = tt_recursive_sketch(Float64, A, 16; seed=seed, block_rks=b, orthogonal=true)
        brv = ones(Int, N+1); brv[1:N] .= b
        for k=N:-1:1; brv[k] = min(brv[k], dims[k]*brv[k+1]); end
        p = skref .÷ brv
        g1 = build_kron_group(A, b, 0, p; seed=seed, orthogonal=true)
        half = copy(p)
        for k=1:N
            lo = k == 1 ? 1 : half[k-1]
            half[k] = max(lo, p[k]-2)
        end
        g2 = KroneckerSketchGroup(Float64, A, b, 0)
        extend_kronecker_sketch!(g2, A, half; seed=seed, orthogonal=true)
        extend_kronecker_sketch!(g2, A, p;    seed=seed, orthogonal=true)
        for l in 1:N+1
            @test finalize_cols_kronecker([g1], l, prks2(l)) ==
                  finalize_cols_kronecker([g2], l, prks2(l))
        end
    end

    @testset "unbiased: E[‖S(y₁⊙y₂)‖²/‖y₁⊙y₂‖²] ≈ 1" begin
        A = (ya, yb)
        prod_dense = ttv_to_tensor(ya) .* ttv_to_tensor(yb)
        np = norm(prod_dense)
        b1, b2 = 6, 3
        brv1 = ones(Int,N+1); brv1[1:N] .= b1; for k=N:-1:1; brv1[k]=min(brv1[k],dims[k]*brv1[k+1]); end
        brv2 = ones(Int,N+1); brv2[1:N] .= b2; for k=N:-1:1; brv2[k]=min(brv2[k],dims[k]*brv2[k+1]); end
        _,sk1 = tt_recursive_sketch(Float64, A, 16; seed=seed, block_rks=b1); pinit = sk1 .÷ brv1
        _,sk2 = tt_recursive_sketch(Float64, A, 12; seed=seed, block_rks=b2); pext  = sk2 .÷ brv2
        acc = 0.0; nseed = 300
        for s in 1:nseed
            gi = build_kron_group(A, b1, 0, pinit; seed=s, orthogonal=true)
            ge = build_kron_group(A, b2, 1, pext;  seed=s, orthogonal=true)
            comb1 = finalize_cols_kronecker([gi, ge], 1, prks2(1); weighting=:equal)
            acc += (norm(comb1)/np)^2
        end
        @test isapprox(acc/nseed, 1.0; atol=0.06)
    end

    @testset "KroneckerCachedSketch: grow-then-grow-more == from-scratch (bit-identical)" begin
        A = (ya, yb, yc)
        for (brk, brk_inc) in ((4, 4), (6, 2))     # uniform and mixed block_rks
            want1 = fill(16, N+1); want1[N+1] = 1
            want2 = fill(30, N+1); want2[N+1] = 1
            c1 = cached_kronecker_sketch(Float64, A, brk, brk_inc, 10; seed=seed)
            ensure_columns!(c1, A, want1; seed=seed)
            ensure_columns!(c1, A, want2; seed=seed)
            c2 = cached_kronecker_sketch(Float64, A, brk, brk_inc, 10; seed=seed)
            ensure_columns!(c2, A, want2; seed=seed)
            for l in 1:N+1
                @test sketch_matrix(c1, l, prks3(l)) == sketch_matrix(c2, l, prks3(l))
            end
            @test all(size(sketch_matrix(c1, l, prks3(l)), 2) >= want2[l] for l in 1:N)
        end
    end

    @testset "forward sweep (reverse=false)" begin
        fbrv(b) = (v = ones(Int,N+1); v[2:N+1] .= b; for k=1:N; v[k+1] = min(v[k+1], dims[k]*v[k]); end; v)
        @testset "single group == tt_recursive_sketch(tuple; reverse=false) (bit-identical)" begin
            for (A, prks) in (((ya, yb), prks2), ((ya, yb, yc), prks3)), b in (3, 6)
                Wref, skref = tt_recursive_sketch(Float64, A, 12; seed=seed, block_rks=b, orthogonal=true, reverse=false)
                p = skref .÷ fbrv(b)
                g = KroneckerSketchGroup(Float64, A, b, 0; reverse=false)
                extend_kronecker_sketch!(g, A, p; seed=seed, orthogonal=true)
                for l in 1:N+1
                    @test finalize_cols_kronecker([g], l, prks(l)) == reshape(Wref[l], prks(l), :)
                end
            end
        end
        @testset "cache reverse=false: grow-then-grow == from-scratch (bit-identical)" begin
            A = (ya, yb, yc)
            for (brk, brk_inc) in ((4, 4), (6, 2))
                want1 = fill(16, N+1); want1[1] = 1
                want2 = fill(28, N+1); want2[1] = 1
                c1 = cached_kronecker_sketch(Float64, A, brk, brk_inc, 10; reverse=false, seed=seed)
                ensure_columns!(c1, A, want1; seed=seed)
                ensure_columns!(c1, A, want2; seed=seed)
                c2 = cached_kronecker_sketch(Float64, A, brk, brk_inc, 10; reverse=false, seed=seed)
                ensure_columns!(c2, A, want2; seed=seed)
                for l in 1:N+1
                    @test sketch_matrix(c1, l, prks3(l)) == sketch_matrix(c2, l, prks3(l))
                end
                @test all(size(sketch_matrix(c1, l, prks3(l)), 2) >= want2[l] for l in 2:N+1)
            end
        end
    end
end
