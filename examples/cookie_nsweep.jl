using CairoMakie
using Serialization
using Printf

# Reuse the operator/import/GMRES (cookie_gmres.jl) plus `make_sumround_krp` / `run_variant`
# / `DET_REF` from the block-rank comparison. Neither include runs its own main on `include`.
include(joinpath(@__DIR__, "cookie_blockrks_compare.jl"))

"""
Parameter-sample sweep for the cookie TT-GMRES: vary n (= number of parameter samples per
parametric mode) over the paper's range and track time / rank / accuracy for a few Sum+Round
sketch variants (Al Daas et al., arXiv:2511.03598, §4.3).

The randomized solves are essentially n-independent in iteration count and solution rank (set by
the 7-cookie geometry, not by n); what grows with n is the parameter-mode core size. The single
deterministic point (n=8) is shown for reference — the naive baseline is ~40× slower and grows
with n (the paper reports out-of-memory at n=128/256).
"""

const SWEEP_VARIANTS = [
    (label = "KRP",            block_rks = 1, orthogonal = false),  # strict Khatri–Rao (faithful)
    (label = "TTStack-N",      block_rks = 8, orthogonal = false),  # block_rks=N, Gaussian (winner)
    (label = "orth-TTStack-N", block_rks = 8, orthogonal = true),   # confirm orth stays a wash
]

const NS = [8, 16, 32, 64, 128, 256]

const _SW_COLORS = Dict("KRP" => :tomato, "TTStack-N" => :seagreen, "orth-TTStack-N" => :royalblue)
const _SW_MARKERS = Dict("KRP" => :circle, "TTStack-N" => :rect, "orth-TTStack-N" => :utriangle)

function run_cookie_nsweep(; ns = NS, dir = DATA_DIR, minD = 1.0, maxD = 5.0,
                           tol = 1e-8, maxit = 50, seed = 1234,
                           outdir = joinpath(@__DIR__, "..", "out", "cookie_nsweep"))
    mkpath(outdir)
    println("Loading n-independent spatial matrices + factorizing preconditioner once …")
    A0, A_c, a0 = load_cookie_data(dir)      # spatial stiffness matrices do not depend on n
    prec = make_preconditioner(A0, A_c)      # mean spatial operator — n-independent

    # results[label] = Dict(:n, :iters, :time, :t_sr, :relres, :peak, :final)
    results = Dict(v.label => Dict(:n => Int[], :iters => Int[], :time => Float64[],
                                   :t_sr => Float64[], :relres => Float64[],
                                   :peak => Int[], :final => Int[]) for v in SWEEP_VARIANTS)
    warmed = false

    for n in ns
        println("\n########## n = $n parameter samples ##########")
        op = CookieOp(A0, A_c, n; minD = minD, maxD = maxD)
        b = (1 / norm(cookie_rhs(op, a0))) * cookie_rhs(op, a0)
        apply_op = x -> apply_summands(op, x)
        if !warmed
            print("  warming up … "); flush(stdout)
            _ = run_variant(apply_op, b, prec,
                            make_sumround_krp(block_rks = 8, orthogonal = true, seed = seed);
                            tol = tol, maxit = 2)
            println("done"); warmed = true
        end
        for v in SWEEP_VARIANTS
            @printf("  %-16s (block_rks=%d, orth=%s): ", v.label, v.block_rks, v.orthogonal)
            flush(stdout)
            sumround = make_sumround_krp(block_rks = v.block_rks, orthogonal = v.orthogonal, seed = seed)
            try
                res = run_variant(apply_op, b, prec, sumround; tol = tol, maxit = maxit)
                d = results[v.label]
                push!(d[:n], n); push!(d[:iters], res.iters); push!(d[:time], res.time)
                push!(d[:t_sr], res.t_sr); push!(d[:relres], res.relres)
                push!(d[:peak], res.peak); push!(d[:final], res.final)
                @printf("%d it, %.1fs (S+R %.1fs), relres=%.2e, peak rk=%d, final rk=%d\n",
                        res.iters, res.time, res.t_sr, res.relres, res.peak, res.final)
            catch err
                if err isa OutOfMemoryError
                    @printf("OUT OF MEMORY — skipping\n")
                else
                    @printf("FAILED (%s) — skipping\n", typeof(err))
                end
            end
            flush(stdout)
            serialize(joinpath(outdir, "cookie_nsweep_results.jls"), results)  # incremental
        end
    end

    print_nsweep_table(results)
    plot_cookie_nsweep(results; outdir = outdir)
    return results
end

function print_nsweep_table(results)
    println("\n", repeat('=', 78))
    @printf("%-16s | %4s | %5s | %9s | %11s | %9s | %9s\n",
            "variant", "n", "iters", "wall (s)", "true relres", "peak rk", "final rk")
    println(repeat('-', 78))
    for v in SWEEP_VARIANTS
        d = results[v.label]
        for i in eachindex(d[:n])
            @printf("%-16s | %4d | %5d | %9.1f | %11.2e | %9d | %9d\n",
                    v.label, d[:n][i], d[:iters][i], d[:time][i], d[:relres][i], d[:peak][i], d[:final][i])
        end
    end
    @printf("%-16s | %4d | %5d | %9.1f | %11.2e | %9d | %9d   (reference)\n",
            DET_REF.label, 8, DET_REF.iters, DET_REF.time, DET_REF.relres, DET_REF.peak, DET_REF.final)
    println(repeat('=', 78))
end

function plot_cookie_nsweep(results; outdir)
    CairoMakie.activate!(type = "pdf")
    fig = Figure(size = (1300, 540))
    Label(fig[0, 1:2], "Cookie TT-GMRES — parameter-sample sweep (7 cookies, n1=1681)";
          fontsize = 14, tellwidth = false)

    ax1 = Axis(fig[1, 1], xlabel = "parameter samples n (per mode)", ylabel = "wall time (s)",
               xscale = log10, yscale = log10, title = "wall & Sum+Round time vs n", titlesize = 12)
    for v in SWEEP_VARIANTS
        d = results[v.label]; isempty(d[:n]) && continue
        c = _SW_COLORS[v.label]; m = _SW_MARKERS[v.label]
        scatterlines!(ax1, d[:n], d[:time]; color = c, marker = m, markersize = 9,
                      linewidth = 2, label = "$(v.label) wall")
        scatterlines!(ax1, d[:n], d[:t_sr]; color = c, marker = m, markersize = 7,
                      linewidth = 1, linestyle = :dash)
    end
    scatter!(ax1, [8], [DET_REF.time]; color = :black, marker = :star5, markersize = 16,
             label = "det (n=8 ref)")
    axislegend(ax1; position = :lt, framevisible = true, labelsize = 9)

    ax2 = Axis(fig[1, 2], xlabel = "parameter samples n (per mode)", ylabel = "TT-rank",
               xscale = log10, title = "peak (solid) & final (dotted) TT-rank vs n", titlesize = 12)
    for v in SWEEP_VARIANTS
        d = results[v.label]; isempty(d[:n]) && continue
        c = _SW_COLORS[v.label]; m = _SW_MARKERS[v.label]
        scatterlines!(ax2, d[:n], d[:peak]; color = c, marker = m, markersize = 9,
                      linewidth = 2, label = v.label)
        scatterlines!(ax2, d[:n], d[:final]; color = c, marker = m, markersize = 7,
                      linewidth = 1, linestyle = :dot)
    end
    scatter!(ax2, [8], [DET_REF.peak]; color = :black, marker = :star5, markersize = 16,
             label = "det peak (n=8)")
    scatter!(ax2, [8], [DET_REF.final]; color = :gray, marker = :star4, markersize = 12)
    axislegend(ax2; position = :lt, framevisible = true, labelsize = 9)

    fname = joinpath(outdir, "cookie_nsweep.pdf")
    save(fname, fig)
    println("\n→ saved $fname")
    return fig
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_cookie_nsweep()
end
