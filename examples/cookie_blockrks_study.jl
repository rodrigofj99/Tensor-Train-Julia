using CairoMakie
using Serialization
using Printf

# Reuse cookie data/operator/preconditioner, the Sum+Round GMRES + KRP closures (run_variant,
# make_sumround_krp), the sketched_gmres driver, and true_cookie_relres. None run their main on include.
include(joinpath(@__DIR__, "cookie_sketched_gmres.jl"))
include(joinpath(@__DIR__, "sweep_common.jl"))

"""
Study C — cookie parametric-PDE linear system (paper §4.3 / Figs 7–8), block-rank study with BOTH
solvers side by side:
  (a) faithful **Sum+Round TT-GMRES** (`tt_gmres_cookie`, the paper's method: adaptive randomized
      rounding of the Krylov-vector linear combination inside GMRES), and
  (b) the repo's **sketched_gmres** (Nakatsukasa–Tropp, with window-sketch caching).

The named-method axis varies `block_rks` / `orthogonal` (strict-KRP=block_rks-1 raw Gaussian …
TTStack=block_rks≥N orthogonalized). A *real* deterministic baseline is computed once at n=8 and
cached (`det_baseline.jls`) — replacing the hardcoded `DET_REF` in cookie_blockrks_compare.jl.

NOTE: the linear-combination overload the Sum+Round path uses exposes only `block_rks` (no
`block_rks_inc`), so this study sweeps `block_rks × orthogonal`; the `block_rks_inc` axis is covered
on the single-TT studies (Matérn/synthetic/embedding).

Run:  <julia> --project=. examples/cookie_blockrks_study.jl          # full (slow: det baseline ~90min)
      <julia> --project=. examples/cookie_blockrks_study.jl smoke    # quick check (no det baseline)
"""

const OUTDIR = joinpath(@__DIR__, "..", "out", "cookie_blockrks_study")

# Methods: (label, block_rks, orthogonal). block_rks=1 ⇒ KRP; ≥N ⇒ TTStack.
cookie_methods() = [(label="strict-KRP", blk=1,  orth=false),
                    (label="orth-KRP",   blk=1,  orth=true),
                    (label="TTStack-N",  blk=8,  orth=true),
                    (label="TTStack-2N", blk=16, orth=true)]

# Deterministic Sum+Round closure (rounds the Krylov linear combination exactly).
sumround_det() = (ys, α, ε) -> tt_rounding(weighted_sum(collect(Float64, α), ys); tol = ε)

"""Compute (once) and cache the deterministic TT-GMRES baseline at n=8."""
function load_or_compute_det_baseline(apply_op, b, prec; tol, maxit, outdir)
    path = joinpath(outdir, "det_baseline.jls")
    if isfile(path)
        println("  loading cached deterministic baseline from $path"); return deserialize(path)
    end
    println("  computing deterministic baseline (this is the ~90 min solve) …"); flush(stdout)
    r = run_variant(apply_op, b, prec, sumround_det(); tol = tol, maxit = maxit)
    mkpath(outdir); serialize(path, r)
    return r
end

function run_cookie_study(; n = 8, dir = DATA_DIR, minD = 1.0, maxD = 5.0, tol = 1e-8, maxit = 50,
                          sketched_tol = 1e-7, seed = 1234, smoke = false,
                          outdir = OUTDIR, with_det = true, methods = cookie_methods())
    smoke && (maxit = 3; with_det = false; methods = methods[[1, 3]])  # strict-KRP + TTStack-N
    mkpath(outdir)
    println("Loading cookie data / operator / preconditioner (n=$n) …")
    A0, A_c, a0 = load_cookie_data(dir)
    op = CookieOp(A0, A_c, n; minD = minD, maxD = maxD)
    b = (1 / norm(cookie_rhs(op, a0))) * cookie_rhs(op, a0)
    prec = make_preconditioner(A0, A_c)
    apply_op = x -> apply_summands(op, x)
    d = length(op.dims)
    x0 = zeros_tt(Float64, op.dims, ones(Int, d + 1))
    println("  modes=$d, n1=$(op.dims[1]), params∈[$minD,$maxD]")

    # Warmup both solver paths (runtime block_rks/orth ⇒ one warmup compiles each).
    print("  warming up … "); flush(stdout)
    _ = run_variant(apply_op, b, prec, make_sumround_krp(block_rks=8, orthogonal=true, seed=seed); tol=tol, maxit=2)
    _ = sketched_gmres(apply_op, b, x0; prec=prec, m=4, tol=tol, max_iters=4, k_trunc=4,
                       block_rks=8, seed=seed, verbose=false)
    println("done\n")

    det = with_det ? load_or_compute_det_baseline(apply_op, b, prec; tol=tol, maxit=maxit, outdir=outdir) : nothing
    det_wall = det === nothing ? NaN : det.time

    results = Dict{String,Any}()        # label => (sr=…, sk=…)
    order = String[]
    for v in methods
        @printf("=== %-12s (block_rks=%d, orth=%s) ===\n", v.label, v.blk, v.orth); flush(stdout)
        # (a) Sum+Round TT-GMRES
        sr_round = make_sumround_krp(block_rks=v.blk, orthogonal=v.orth, seed=seed)
        sr = run_variant(apply_op, b, prec, sr_round; tol=tol, maxit=maxit)
        @printf("   Sum+Round : %2d its, %7.1fs (S+R %6.1fs), relres=%.2e, peak rk=%d, final rk=%d, speedup=%.2f\n",
                sr.iters, sr.time, sr.t_sr, sr.relres, sr.peak, sr.final, det_wall/sr.time); flush(stdout)
        # (b) sketched_gmres (window-sketch caching on)
        t_sk = @elapsed (xsk, hsk) = sketched_gmres(apply_op, b, x0; prec=prec, m=maxit, tol=sketched_tol,
                                                    max_iters=maxit, k_trunc=4, block_rks=v.blk,
                                                    seed=seed, reuse_sketches=true, verbose=false)
        relres_sk = true_cookie_relres(op, xsk, b)
        sk = (iters=length(hsk), time=t_sk, relres=relres_sk, final=maximum(xsk.ttv_rks))
        @printf("   sketched  : %2d its, %7.1fs,                  relres=%.2e,            final rk=%d, speedup=%.2f\n",
                sk.iters, sk.time, sk.relres, sk.final, det_wall/sk.time); flush(stdout)
        results[v.label] = (sr=sr, sk=sk)
        push!(order, v.label)
    end

    serialize(joinpath(outdir, "cookie_study_results.jls"), (order=order, results=results, det=det))
    print_study_table(order, results, det)
    plot_cookie_study(order, results, det; outdir=outdir, n=n)
    return order, results, det
end

function print_study_table(order, results, det)
    println("\n", repeat('=', 104))
    @printf("%-12s | %-9s | %5s | %9s | %11s | %11s | %8s | %8s | %7s\n",
            "method", "solver", "iters", "wall (s)", "S+R (s)", "true relres", "peak rk", "final rk", "speedup")
    println(repeat('-', 104))
    dw = det === nothing ? NaN : det.time
    for lbl in order
        r = results[lbl]
        @printf("%-12s | %-9s | %5d | %9.1f | %11.1f | %11.2e | %8d | %8d | %7.2f\n",
                lbl, "Sum+Round", r.sr.iters, r.sr.time, r.sr.t_sr, r.sr.relres, r.sr.peak, r.sr.final, dw/r.sr.time)
        @printf("%-12s | %-9s | %5d | %9.1f | %11s | %11.2e | %8s | %8d | %7.2f\n",
                "", "sketched", r.sk.iters, r.sk.time, "—", r.sk.relres, "—", r.sk.final, dw/r.sk.time)
    end
    if det !== nothing
        @printf("%-12s | %-9s | %5d | %9.1f | %11s | %11.2e | %8d | %8d | %7s\n",
                "det", "Sum+Round", det.iters, det.time, "—", det.relres, det.peak, det.final, "1.00")
    end
    println(repeat('=', 104))
end

function plot_cookie_study(order, results, det; outdir, n)
    CairoMakie.activate!(type="pdf"); mkpath(outdir)
    fig = Figure(size=(1300, 540))
    Label(fig[0, 1:2], "Cookie TT-GMRES block-rank study (n=$n) — Sum+Round rank growth & both-solver timing";
          fontsize=13, tellwidth=false)

    # Panel (a): max TT-rank of the Arnoldi basis per GMRES iteration (Sum+Round path; paper Fig 7).
    ax1 = Axis(fig[1, 1], xlabel="GMRES iteration", ylabel="max TT-rank of basis vector",
               title="Sum+Round rank growth per iteration", titlesize=12)
    for lbl in order
        rk = results[lbl].sr.rk; c, m = variant_style(lbl)
        scatterlines!(ax1, 1:length(rk), rk; color=c, linewidth=2, marker=m, markersize=7, label=lbl)
    end
    if det !== nothing
        scatterlines!(ax1, 1:length(det.rk), det.rk; color=:black, linewidth=2, linestyle=:dash, label="det")
    end
    axislegend(ax1; position=:rb, framevisible=true, labelsize=10)

    # Panel (b): wall time per method, both solvers.
    nv = length(order)
    ax2 = Axis(fig[1, 2], xlabel="method", ylabel="wall time (s)", yscale=log10,
               title="wall time: Sum+Round vs sketched", titlesize=12,
               xticks=(1:nv, order), xticklabelrotation=π/6)
    srw = [results[l].sr.time for l in order]; skw = [results[l].sk.time for l in order]
    scatterlines!(ax2, 1:nv, srw; color=:crimson, marker=:rect, markersize=11, linewidth=1, label="Sum+Round")
    scatterlines!(ax2, 1:nv, skw; color=:royalblue, marker=:circle, markersize=11, linewidth=1, label="sketched")
    if det !== nothing
        hlines!(ax2, [det.time]; color=:gray, linestyle=:dot, linewidth=2)
        text!(ax2, 0.6, det.time; text="det wall ≈ $(round(Int, det.time))s", align=(:left,:bottom), color=:gray, fontsize=10)
    end
    axislegend(ax2; position=:rc, framevisible=true, labelsize=10)

    f = joinpath(outdir, "cookie_blockrks_study.pdf"); save(f, fig); println("\n→ saved $f"); return fig
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_cookie_study(smoke = ("smoke" in ARGS))
end
