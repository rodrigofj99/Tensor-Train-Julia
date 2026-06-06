using CairoMakie
using Serialization
using Printf

# Reuse the faithful cookie TT-GMRES, operator, importer, and Sum+Round helpers.
# (cookie_gmres.jl's PROGRAM_FILE guard keeps its smoke test from running on include.)
include(joinpath(@__DIR__, "cookie_gmres.jl"))

"""
Block-rank / orthogonalized TTStack Sum+Round comparison for the cookie TT-GMRES.

The cookie KRP variant (block_rks=1, orthogonal=false — strict Khatri–Rao) converges fast but
its intermediate TT-ranks balloon (basis vectors reach ~988 vs the deterministic 720), a symptom
of high variance in the sketch-based per-bond norm estimate (Al Daas et al., arXiv:2511.03598,
Remark 3.1). Here we hold everything fixed except the two sketch knobs exposed by the
linear-combination overload `ttrand_rounding_adaptive(α, ys, ε; block_rks, orthogonal)` —
`orthogonal` (within-block QR / TTStack vs raw Gaussian) and `block_rks` (sketch block rank) —
and measure whether higher block rank / orthogonalized sketches tame the ranks (and time) while
keeping the same ~1e-7 true residual.
"""

# Variants: only block_rks and orthogonal vary; ℓ_min/ℓ_inc/n_samples/seed fixed below.
const VARIANTS = [
    (label = "strict-KRP",      block_rks = 1,  orthogonal = false),  # current cookie default
    (label = "orth-KRP",        block_rks = 1,  orthogonal = true),
    (label = "TTStack-N",       block_rks = 8,  orthogonal = true),   # overload default
    (label = "TTStack-N-Gauss", block_rks = 8,  orthogonal = false),
    (label = "TTStack-16",      block_rks = 16, orthogonal = true),
    (label = "TTStack-32",      block_rks = 32, orthogonal = true),
]

# Deterministic baseline measured earlier this session (no re-run). Per-iteration max rank of the
# (rounded) Arnoldi basis vectors, then final/peak/accuracy after the n=8 solve.
const DET_REF = (label = "det (ref)",
                 iters = 16, time = 5367.9, relres = 3.641e-7,
                 peak = 720, final = 150,
                 rk = [7, 35, 119, 203, 336, 465, 576, 656,
                       696, 717, 720, 713, 689, 673, 647, 621])

"""Closure factory for the KRP/TTStack Sum+Round with configurable sketch knobs."""
function make_sumround_krp(; block_rks::Int, orthogonal::Bool, n_samples::Int = 20,
                           seed::Int = 1234, final_round::Bool = true)
    return function (ys, α, ε)
        y = ttrand_rounding_adaptive(collect(Float64, α), ys, ε;
                                     block_rks = block_rks, orthogonal = orthogonal,
                                     n_samples = n_samples, seed = seed)
        # Final deterministic rounding pass to shed excess rank from the adaptive sketch
        # (Al Daas et al., arXiv:2511.03598, Remark 3.2).
        return final_round ? tt_rounding(y; tol = ε) : y
    end
end

"""Run one GMRES solve and collect the comparison metrics."""
function run_variant(apply_op, b, prec, sumround; tol, maxit)
    tels = @elapsed (x, res_hist, rank_hist, t_sr) =
        tt_gmres_cookie(apply_op, b, prec, sumround; tol = tol, maxit = maxit, verbose = false)
    # True relative residual the MATLAB way: r = Σ_j A_j x − b.
    summ = apply_op(x)
    r = tt_rounding(weighted_sum(vcat(ones(length(summ)), -1.0), vcat(summ, [b])); tol = tol * 1e-2)
    relres = norm(r) / norm(b)
    return (iters = length(res_hist), time = tels, t_sr = t_sr, relres = relres,
            peak = maximum(rank_hist), final = maximum(x.ttv_rks), rk = rank_hist)
end

function compare_cookie_variants(; n = 8, dir = DATA_DIR, minD = 1.0, maxD = 5.0,
                                 tol = 1e-8, maxit = 50, seed = 1234,
                                 outdir = joinpath(@__DIR__, "..", "out", "cookie_blockrks"))
    mkpath(outdir)
    println("Loading cookie data and building operator/RHS/preconditioner (n=$n) …")
    A0, A_c, a0 = load_cookie_data(dir)
    op = CookieOp(A0, A_c, n; minD = minD, maxD = maxD)
    b = (1 / norm(cookie_rhs(op, a0))) * cookie_rhs(op, a0)
    prec = make_preconditioner(A0, A_c)
    apply_op = x -> apply_summands(op, x)
    println("  modes=$(length(op.dims)), n1=$(op.dims[1]), params∈[$minD,$maxD]")

    # Warmup: compile the GMRES + Sum+Round path so the first variant isn't penalised
    # (orthogonal/block_rks are runtime values → one warmup covers all variants).
    print("  warming up … "); flush(stdout)
    _ = run_variant(apply_op, b, prec,
                    make_sumround_krp(block_rks = 8, orthogonal = true, seed = seed);
                    tol = tol, maxit = 2)
    println("done\n")

    results = Dict{String,Any}()
    order = String[]
    for v in VARIANTS
        @printf("--- %-16s (block_rks=%2d, orthogonal=%s) ---\n", v.label, v.block_rks, v.orthogonal)
        flush(stdout)
        sumround = make_sumround_krp(block_rks = v.block_rks, orthogonal = v.orthogonal, seed = seed)
        res = run_variant(apply_op, b, prec, sumround; tol = tol, maxit = maxit)
        @printf("    %d iters, %.1fs (Sum+Round %.1fs), true relres=%.3e, peak rank=%d, final rank=%d\n\n",
                res.iters, res.time, res.t_sr, res.relres, res.peak, res.final)
        flush(stdout)
        results[v.label] = res
        push!(order, v.label)
    end

    print_table(order, results)
    serialize(joinpath(outdir, "cookie_blockrks_results.jls"), (order = order, results = results, det = DET_REF))
    plot_cookie_variants(order, results; outdir = outdir)
    return order, results
end

function print_table(order, results)
    println("\n", repeat('=', 92))
    @printf("%-16s | %5s | %9s | %11s | %10s | %9s | %9s\n",
            "variant", "iters", "wall (s)", "Sum+Rnd (s)", "true rel", "peak rk", "final rk")
    println(repeat('-', 92))
    for lbl in order
        r = results[lbl]
        @printf("%-16s | %5d | %9.1f | %11.1f | %10.2e | %9d | %9d\n",
                lbl, r.iters, r.time, r.t_sr, r.relres, r.peak, r.final)
    end
    @printf("%-16s | %5d | %9.1f | %11s | %10.2e | %9d | %9d\n",
            DET_REF.label, DET_REF.iters, DET_REF.time, "~wall", DET_REF.relres, DET_REF.peak, DET_REF.final)
    println(repeat('=', 92))
end

const _COLORS = [:tomato, :orange, :seagreen, :purple, :royalblue, :firebrick]

function plot_cookie_variants(order, results; outdir)
    CairoMakie.activate!(type = "pdf")
    fig = Figure(size = (1300, 540))
    Label(fig[0, 1:2], "Cookie TT-GMRES — block-rank / orthogonalized TTStack Sum+Round (n=8, 1 trial)";
          fontsize = 14, tellwidth = false)

    # Panel (a): max TT-rank of the Arnoldi basis vs GMRES iteration.
    ax1 = Axis(fig[1, 1], xlabel = "GMRES iteration", ylabel = "max TT-rank of basis vector",
               title = "rank growth per iteration", titlesize = 12)
    for (i, lbl) in enumerate(order)
        rk = results[lbl].rk
        scatterlines!(ax1, 1:length(rk), rk; color = _COLORS[mod1(i, length(_COLORS))],
                      linewidth = 2, markersize = 7, label = lbl)
    end
    lines!(ax1, 1:length(DET_REF.rk), DET_REF.rk; color = :black, linewidth = 2,
           linestyle = :dash, label = DET_REF.label)
    axislegend(ax1; position = :rb, framevisible = true, labelsize = 10)

    # Panel (b): wall time and Sum+Round time per variant (log scale).
    nv = length(order)
    ax2 = Axis(fig[1, 2], xlabel = "variant", ylabel = "time (s)", yscale = log10,
               title = "wall vs Sum+Round time", titlesize = 12,
               xticks = (1:nv, order), xticklabelrotation = π / 6)
    walls = [results[lbl].time for lbl in order]
    srs   = [results[lbl].t_sr for lbl in order]
    scatterlines!(ax2, 1:nv, walls; color = :black, marker = :rect, markersize = 11,
                  linewidth = 1, label = "wall time")
    scatterlines!(ax2, 1:nv, srs; color = :crimson, marker = :circle, markersize = 11,
                  linewidth = 1, label = "Sum+Round time")
    hlines!(ax2, [DET_REF.time]; color = :gray, linestyle = :dot, linewidth = 2)
    text!(ax2, 0.6, DET_REF.time; text = "det wall ≈ $(round(Int, DET_REF.time))s",
          align = (:left, :bottom), color = :gray, fontsize = 10)
    axislegend(ax2; position = :rc, framevisible = true, labelsize = 10)

    fname = joinpath(outdir, "cookie_blockrks_compare.pdf")
    save(fname, fig)
    println("\n→ saved $fname")
    return fig
end

if abspath(PROGRAM_FILE) == @__FILE__
    compare_cookie_variants()
end
