using Printf

# Reuse the cookie data/operator/preconditioner + the Sum+Round GMRES and KRP closures.
# (cookie_blockrks_compare.jl includes cookie_gmres.jl; neither runs its main on include.)
include(joinpath(@__DIR__, "cookie_blockrks_compare.jl"))

"""
Sketched TT-GMRES (Nakatsukasa–Tropp, arXiv:2111.00113) on the cookie problem, compared against
the Sum+Round KRP TT-GMRES. Uses the package's `sketched_gmres` with the cookie operator assembled
as a single `TToperator` (`cookie_operator_tto`) and the mode-1 LU as a right preconditioner.
"""

"""True relative residual ‖Σ_j A_j x − b‖ / ‖b‖, recomputed the MATLAB way."""
true_cookie_relres(op::CookieOp, x::TTvector, b::TTvector) =
    norm(tt_rounding(weighted_sum(vcat(ones(length(op.spatial)), -1.0),
                                  vcat(apply_summands(op, x), [b])); tol = 1e-10)) / norm(b)

function run_sketched_cookie(; n = 8, minD = 1.0, maxD = 5.0, tol = 1e-7, m = 24,
                             max_iters = 24, k_trunc = 4, ε_cap = 0.1, rmax = 1024, seed = 1234,
                             dir = DATA_DIR)
    A0, A_c, a0 = load_cookie_data(dir)
    op   = CookieOp(A0, A_c, n; minD = minD, maxD = maxD)
    b    = (1 / norm(cookie_rhs(op, a0))) * cookie_rhs(op, a0)
    prec = make_preconditioner(A0, A_c)
    d    = length(op.dims)
    x0   = zeros_tt(Float64, op.dims, ones(Int, d + 1))

    println("\n=== Sketched TT-GMRES (cookie n=$n, modes=$d, n1=$(op.dims[1]), " *
            "max_iters=$max_iters (s=$(2max_iters)), k_trunc=$k_trunc, ε_cap=$ε_cap) ===")
    # Operator as the cheap rank-preserving Kronecker-summand list (no dense TToperator).
    apply_op = x -> apply_summands(op, x)
    t_s = @elapsed (xs, hist) = sketched_gmres(apply_op, b, x0; prec = prec, m = m, tol = tol,
                                               rmax = rmax, max_iters = max_iters,
                                               k_trunc = k_trunc, ε_cap = ε_cap, block_rks = 8,
                                               seed = seed, verbose = true)
    relres_s = true_cookie_relres(op, xs, b)
    @printf("→ sketched GMRES: %d its, %.1fs, TRUE relres=%.3e, final rank=%d\n",
            length(hist), t_s, relres_s, maximum(xs.ttv_rks))

    # Sum+Round KRP baseline (block_rks=8, Gaussian, final deterministic pass).
    println("\n--- Sum+Round KRP baseline (block_rks=8, final pass) ---")
    sr = make_sumround_krp(block_rks = 8, orthogonal = false, seed = seed, final_round = true)
    t_b = @elapsed (xb, resb, rkb, t_sr) = tt_gmres_cookie(apply_op, b, prec, sr;
                                                           tol = tol, maxit = 50, verbose = false)
    relres_b = true_cookie_relres(op, xb, b)
    @printf("→ Sum+Round KRP: %d its, %.1fs (S+R %.1fs), TRUE relres=%.3e, peak rank=%d, final rank=%d\n",
            length(resb), t_b, t_sr, relres_b, maximum(rkb), maximum(xb.ttv_rks))

    println("\n", repeat('=', 72))
    @printf("%-18s | %5s | %9s | %11s | %9s\n", "method", "iters", "wall (s)", "true relres", "final rk")
    println(repeat('-', 72))
    @printf("%-18s | %5d | %9.1f | %11.2e | %9d\n", "sketched GMRES", length(hist), t_s, relres_s, maximum(xs.ttv_rks))
    @printf("%-18s | %5d | %9.1f | %11.2e | %9d\n", "Sum+Round KRP", length(resb), t_b, relres_b, maximum(xb.ttv_rks))
    println(repeat('=', 72))
    return (xs = xs, hist = hist, relres_s = relres_s, xb = xb, relres_b = relres_b)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_sketched_cookie()
end
