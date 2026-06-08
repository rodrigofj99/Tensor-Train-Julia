using TensorTrains, Test, LinearAlgebra
using TensorTrains: OperatorSketchGroup, extend_operator_sketch!, finalize_cols_operator,
                    tt_recursive_sketch, OperatorCachedSketch, cached_operator_sketch,
                    ensure_columns!, sketch_array,
                    SketchGroup, extend_recursive_sketch!, finalize_cols

BLAS.set_num_threads(1)   # determinism for the bit-identity checks

# Build a single operator group up to per-bond counts `p` (length N+1, non-decreasing over 1:N, p[N+1]=1).
function build_op_group(A, y, block_rks, group, p; seed, orthogonal)
    g = OperatorSketchGroup(Float64, A, y, block_rks, group)
    extend_operator_sketch!(g, A, y, p; seed=seed, orthogonal=orthogonal)
    return g
end

@testset "extend_operator_sketch! / finalize_cols_operator" begin
    dims = (4, 5, 3, 6, 4); N = 5; seed = 909
    y = rand_tt(Float64, dims, 7; seed=11)
    A = rand_tto(dims, 4)                             # TToperator (reused across checks)
    a_rks = y.ttv_rks; op_rks = A.tto_rks

    @testset "single group == tt_recursive_sketch(A,y) (bit-identical)" begin
        for orth in (true, false), b in (3, 6)
            Wref, skref = tt_recursive_sketch(Float64, A, y, 12; seed=seed, block_rks=b, orthogonal=orth)
            brv = ones(Int, N+1); brv[1:N] .= b
            for k=N:-1:1; brv[k] = min(brv[k], dims[k]*brv[k+1]); end
            p = skref .÷ brv
            g = build_op_group(A, y, b, 0, p; seed=seed, orthogonal=orth)
            for l in 1:N+1
                fin = finalize_cols_operator([g], l, a_rks[l], op_rks[l]; weighting=:equal)
                @test fin == Wref[l]            # bit-identical (group 0 ⇒ same blocks)
            end
        end
    end

    @testset "one-shot vs two-shot extension (bit-identical, history-independent)" begin
        b = 4
        Wref, skref = tt_recursive_sketch(Float64, A, y, 16; seed=seed, block_rks=b, orthogonal=true)
        brv = ones(Int, N+1); brv[1:N] .= b
        for k=N:-1:1; brv[k] = min(brv[k], dims[k]*brv[k+1]); end
        p = skref .÷ brv
        g1 = build_op_group(A, y, b, 0, p; seed=seed, orthogonal=true)
        half = copy(p)
        for k=1:N
            lo = k == 1 ? 1 : half[k-1]
            half[k] = max(lo, p[k]-2)
        end
        g2 = OperatorSketchGroup(Float64, A, y, b, 0)
        extend_operator_sketch!(g2, A, y, half; seed=seed, orthogonal=true)
        extend_operator_sketch!(g2, A, y, p;    seed=seed, orthogonal=true)
        for l in 1:N+1
            @test finalize_cols_operator([g1], l, a_rks[l], op_rks[l]) ==
                  finalize_cols_operator([g2], l, a_rks[l], op_rks[l])
        end
    end

    @testset "residual: S(A·y) − S(b) == S(A·y − b) (shared blocks)" begin
        b = rand_tt(Float64, dims, 5; seed=33)
        Ay = A * y                                  # explicit operator-vector product
        Ay_minus_b = Ay - b
        brk = 4
        brv = ones(Int, N+1); brv[1:N] .= brk
        for k=N:-1:1; brv[k] = min(brv[k], dims[k]*brv[k+1]); end
        _, sk = tt_recursive_sketch(Float64, A, y, 14; seed=seed, block_rks=brk)
        p = sk .÷ brv
        gop = build_op_group(A, y, brk, 0, p; seed=seed, orthogonal=true)
        gb  = SketchGroup(Float64, b, brk, 0); extend_recursive_sketch!(gb, b, p; seed=seed, orthogonal=true)
        # boundary l=1: ‖S(Ay) − S(b)‖ ≈ ‖Ay − b‖
        Sop1 = vec(finalize_cols_operator([gop], 1, a_rks[1], op_rks[1]))
        Sb1  = vec(finalize_cols([gb], 1, b.ttv_rks[1]))
        @test isapprox(norm(Sop1 .- Sb1), norm(Ay_minus_b); rtol=0.3)
    end

    @testset "OperatorCachedSketch: grow-then-grow-more == from-scratch (bit-identical)" begin
        for (brk, brk_inc) in ((4, 4), (6, 2))     # uniform and mixed block_rks
            want1 = fill(16, N+1); want1[N+1] = 1
            want2 = fill(30, N+1); want2[N+1] = 1
            c1 = cached_operator_sketch(Float64, A, y, brk, brk_inc, 10; seed=seed)
            ensure_columns!(c1, A, y, want1; seed=seed)
            ensure_columns!(c1, A, y, want2; seed=seed)
            c2 = cached_operator_sketch(Float64, A, y, brk, brk_inc, 10; seed=seed)
            ensure_columns!(c2, A, y, want2; seed=seed)
            for l in 1:N+1
                @test sketch_array(c1, l, a_rks[l], op_rks[l]) == sketch_array(c2, l, a_rks[l], op_rks[l])
            end
            @test all(size(sketch_array(c1, l, a_rks[l], op_rks[l]), 3) >= want2[l] for l in 1:N)
        end
    end
end
