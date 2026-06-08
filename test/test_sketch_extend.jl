using TensorTrains, Test, LinearAlgebra
using TensorTrains: SketchGroup, extend_recursive_sketch!, finalize_cols, tt_recursive_sketch,
                    CachedSketch, cached_sketch, ensure_columns!, sketch_matrix

BLAS.set_num_threads(1)   # determinism for the bit-identity checks

# Build a single group up to per-bond counts `p` (length N+1, non-decreasing over 1:N, p[N+1]=1).
function build_group(A, block_rks, group, p; seed, orthogonal)
    g = SketchGroup(Float64, A, block_rks, group)
    extend_recursive_sketch!(g, A, p; seed=seed, orthogonal=orthogonal)
    return g
end

@testset "extend_recursive_sketch! / finalize_cols" begin
    dims = (8,7,9,6,8); N = 5; seed = 4242
    A = rand_tt(Float64, dims, 9; seed=77)
    rks = A.ttv_rks

    @testset "single group == tt_recursive_sketch (bit-identical)" begin
        for orth in (true, false), b in (3, 7)
            Wref, skref = tt_recursive_sketch(Float64, A, 14; seed=seed, block_rks=b, orthogonal=orth)
            brv = ones(Int, N+1); brv[1:N] .= b
            for k=N:-1:1; brv[k] = min(brv[k], dims[k]*brv[k+1]); end
            p = skref .÷ brv
            g = build_group(A, b, 0, p; seed=seed, orthogonal=orth)
            for l in 1:N+1
                fin = finalize_cols([g], l, rks[l]; weighting=:equal)
                @test fin == Wref[l]            # bit-identical (group 0 ⇒ same blocks as tt_recursive_sketch)
            end
        end
    end

    @testset "one-shot vs two-shot extension (bit-identical, history-independent)" begin
        b = 4
        Wref, skref = tt_recursive_sketch(Float64, A, 18; seed=seed, block_rks=b, orthogonal=true)
        brv = ones(Int, N+1); brv[1:N] .= b
        for k=N:-1:1; brv[k] = min(brv[k], dims[k]*brv[k+1]); end
        p = skref .÷ brv
        # one shot
        g1 = build_group(A, b, 0, p; seed=seed, orthogonal=true)
        # two shots: a smaller non-decreasing partial first (reduce each bond by up to 2 but keep
        # it ≥ the previous bond, so 1:N stays non-decreasing; leave the N+1 boundary at 1), then full.
        half = copy(p)
        for k=1:N
            lo = k == 1 ? 1 : half[k-1]
            half[k] = max(lo, p[k]-2)
        end
        g2 = SketchGroup(Float64, A, b, 0)
        extend_recursive_sketch!(g2, A, half; seed=seed, orthogonal=true)
        extend_recursive_sketch!(g2, A, p;    seed=seed, orthogonal=true)
        for l in 1:N+1
            @test finalize_cols([g1], l, rks[l]) == finalize_cols([g2], l, rks[l])
        end
    end

    @testset "two-group mixed block_rks: unbiased isometry" begin
        b1, b2 = 7, 3
        brv1 = ones(Int,N+1); brv1[1:N] .= b1; for k=N:-1:1; brv1[k]=min(brv1[k],dims[k]*brv1[k+1]); end
        brv2 = ones(Int,N+1); brv2[1:N] .= b2; for k=N:-1:1; brv2[k]=min(brv2[k],dims[k]*brv2[k+1]); end
        _,sk1 = tt_recursive_sketch(Float64, A, 14; seed=seed, block_rks=b1); pinit = sk1 .÷ brv1
        _,sk2 = tt_recursive_sketch(Float64, A, 9;  seed=seed, block_rks=b2); pext  = sk2 .÷ brv2
        ny = norm(A)
        acc = 0.0; nseed = 300
        for s in 1:nseed
            gi = build_group(A, b1, 0, pinit; seed=s, orthogonal=true)
            ge = build_group(A, b2, 1, pext;  seed=s, orthogonal=true)
            comb1 = finalize_cols([gi, ge], 1, rks[1]; weighting=:equal)   # boundary sketch vector
            acc += (norm(comb1)/ny)^2
        end
        @test isapprox(acc/nseed, 1.0; atol=0.05)   # unbiased E[‖Sy‖²/‖y‖²] ≈ 1
    end

    @testset "CachedSketch: grow-then-grow-more == from-scratch (bit-identical)" begin
        for (brk, brk_inc) in ((4, 4), (7, 3))     # uniform and mixed block_rks
            want1 = fill(20, N+1); want1[N+1] = 1
            want2 = fill(40, N+1); want2[N+1] = 1
            # incremental: build, grow to want1, then to want2
            c1 = cached_sketch(Float64, A, brk, brk_inc, 12; seed=seed)
            ensure_columns!(c1, A, want1; seed=seed)
            ensure_columns!(c1, A, want2; seed=seed)
            # from scratch: build, grow straight to want2
            c2 = cached_sketch(Float64, A, brk, brk_inc, 12; seed=seed)
            ensure_columns!(c2, A, want2; seed=seed)
            for l in 1:N+1
                @test sketch_matrix(c1, l, rks[l]) == sketch_matrix(c2, l, rks[l])
            end
            # and each bond actually has ≥ want2 columns
            @test all(size(sketch_matrix(c1, l, rks[l]), 2) >= want2[l] for l in 1:N)
        end
    end

    @testset "single-TT overload: cache on/off bit-identical (two-group mixed block_rks)" begin
        y = rand_tt(Float64, dims, 10; seed=42)
        ε = 1e-2; brk = 7; brk_inc = 2
        r0 = ttrand_rounding_adaptive(y, ε; seed=seed, block_rks=brk, block_rks_inc=brk_inc)
        c  = cached_sketch(Float64, y, brk, brk_inc, 4+max(N÷2,4); seed=seed)
        r1 = ttrand_rounding_adaptive(y, ε; seed=seed, block_rks=brk, block_rks_inc=brk_inc, cache=c)
        r2 = ttrand_rounding_adaptive(y, ε; seed=seed, block_rks=brk, block_rks_inc=brk_inc, cache=c)  # reuse grown
        @test norm(r1 - r0) / norm(r0) < 1e-12     # cache-on == cache-off
        @test norm(r2 - r0) / norm(r0) < 1e-12     # reusing the grown cache == fresh
        @test norm(r0 - y) / norm(y) < 1e-10       # rank captures y here
    end
end
