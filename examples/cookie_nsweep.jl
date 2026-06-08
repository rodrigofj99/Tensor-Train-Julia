using CairoMakie
using Serialization
using Printf

# Reuse the cookie operator/import/GMRES, the Sum+Round closures (run_variant / make_sumround_krp /
# sumround_det), the sketched_gmres driver, and true_cookie_relres. cookie_blockrks_study.jl chains the
# includes (→ cookie_sketched_gmres → cookie_blockrks_compare → cookie_gmres) and runs no main on include.
include(joinpath(@__DIR__, "cookie_blockrks_study.jl"))
include(joinpath(@__DIR__, "sweep_common.jl"))

"""
Parameter-sample sweep for the cookie TT-GMRES (Al Daas et al., arXiv:2511.03598, §4.3): vary n (=
number of parameter samples per parametric mode) and track time / rank / accuracy for **three
solvers at every n**:
  - deterministic Sum+Round TT-GMRES (the baseline; recomputed at each n, OOM-guarded),
  - randomized Sum+Round TT-GMRES (strict-KRP and TTStack-N), and
  - sketched GMRES (TTStack-N, window-sketch caching).

Because the deterministic baseline is recomputed at each n, the speedup panel is a genuine per-n
`t_det(n) / t_method(n)` (not a fixed n=8 reference). The cookie solve is essentially n-independent in
iteration count and solution rank (set by the cookie geometry, not n); what grows with n is the
parameter-mode core size, and the deterministic solve eventually runs out of memory at large n
(reported in the paper) — the guard records that as a missing point.
"""

const NS = [8, 16, 32, 64]

# Sum+Round randomized variants run at every n (label, block_rks, orthogonal).
const SR_VARIANTS = [(label = "strict-KRP (S+R)", block_rks = 1, orthogonal = false),
                     (label = "TTStack-N (S+R)",  block_rks = 8, orthogonal = true)]
# Sketched-GMRES variants run at every n.
const SK_VARIANTS = [(label = "TTStack-N (sk)",   block_rks = 8)]

_emptymetrics() = Dict(:n => Int[], :iters => Int[], :time => Float64[], :relres => Float64[],
                       :peak => Int[], :final => Int[])

function run_cookie_nsweep(; ns = NS, dir = DATA_DIR, minD = 1.0, maxD = 5.0,
                           tol = 1e-8, sketched_tol = 1e-7, maxit = 50, seed = 1234, with_det = true,
                           outdir = joinpath(@__DIR__, "..", "out", "cookie_nsweep"))
    mkpath(outdir)
    println("Loading n-independent spatial matrices + factorizing preconditioner once …")
    A0, A_c, a0 = load_cookie_data(dir)
    prec = make_preconditioner(A0, A_c)

    det = _emptymetrics()
    sr  = Dict(v.label => _emptymetrics() for v in SR_VARIANTS)
    sk  = Dict(v.label => _emptymetrics() for v in SK_VARIANTS)
    warmed = false

    for n in ns
        println("\n########## n = $n parameter samples ##########")
        op = CookieOp(A0, A_c, n; minD = minD, maxD = maxD)
        b  = (1 / norm(cookie_rhs(op, a0))) * cookie_rhs(op, a0)
        apply_op = x -> apply_summands(op, x)
        d  = length(op.dims)
        x0 = zeros_tt(Float64, op.dims, ones(Int, d + 1))
        if !warmed
            print("  warming up … "); flush(stdout)
            _ = run_variant(apply_op, b, prec, make_sumround_krp(block_rks=8, orthogonal=true, seed=seed); tol=tol, maxit=2)
            _ = sketched_gmres(apply_op, b, x0; prec=prec, m=4, tol=tol, max_iters=4, k_trunc=4, block_rks=8, seed=seed, verbose=false)
            println("done"); warmed = true
        end

        # Deterministic baseline at THIS n (OOM-guarded; the paper hits OOM at large n).
        if with_det
            print("  deterministic Sum+Round … "); flush(stdout)
            try
                r = run_variant(apply_op, b, prec, sumround_det(); tol = tol, maxit = maxit)
                push!(det[:n], n); push!(det[:iters], r.iters); push!(det[:time], r.time)
                push!(det[:relres], r.relres); push!(det[:peak], r.peak); push!(det[:final], r.final)
                @printf("%d it, %.1fs, relres=%.2e, peak rk=%d\n", r.iters, r.time, r.relres, r.peak)
            catch err
                println(err isa OutOfMemoryError ? "OUT OF MEMORY — skipping" : "FAILED ($(typeof(err))) — skipping")
            end
            flush(stdout)
            dump_incremental(outdir, "cookie_nsweep_results.jls", (det=det, sr=sr, sk=sk))
        end

        # Randomized Sum+Round variants.
        for v in SR_VARIANTS
            @printf("  %-18s … ", v.label); flush(stdout)
            try
                sround = make_sumround_krp(block_rks=v.block_rks, orthogonal=v.orthogonal, seed=seed)
                r = run_variant(apply_op, b, prec, sround; tol=tol, maxit=maxit)
                d_ = sr[v.label]
                push!(d_[:n], n); push!(d_[:iters], r.iters); push!(d_[:time], r.time)
                push!(d_[:relres], r.relres); push!(d_[:peak], r.peak); push!(d_[:final], r.final)
                @printf("%d it, %.1fs (S+R %.1fs), relres=%.2e, peak rk=%d\n", r.iters, r.time, r.t_sr, r.relres, r.peak)
            catch err
                println(err isa OutOfMemoryError ? "OUT OF MEMORY — skipping" : "FAILED ($(typeof(err))) — skipping")
            end
            flush(stdout)
        end

        # Sketched GMRES variants.
        for v in SK_VARIANTS
            @printf("  %-18s … ", v.label); flush(stdout)
            try
                t = @elapsed (xsk, hsk) = sketched_gmres(apply_op, b, x0; prec=prec, m=maxit, tol=sketched_tol,
                                                         max_iters=maxit, k_trunc=4, block_rks=v.block_rks,
                                                         seed=seed, reuse_sketches=true, verbose=false)
                relres = true_cookie_relres(op, xsk, b)
                d_ = sk[v.label]
                push!(d_[:n], n); push!(d_[:iters], length(hsk)); push!(d_[:time], t)
                push!(d_[:relres], relres); push!(d_[:peak], maximum(xsk.ttv_rks)); push!(d_[:final], maximum(xsk.ttv_rks))
                @printf("%d it, %.1fs, relres=%.2e, final rk=%d\n", length(hsk), t, relres, maximum(xsk.ttv_rks))
            catch err
                println(err isa OutOfMemoryError ? "OUT OF MEMORY — skipping" : "FAILED ($(typeof(err))) — skipping")
            end
            flush(stdout)
        end
        dump_incremental(outdir, "cookie_nsweep_results.jls", (det=det, sr=sr, sk=sk))
    end

    print_nsweep_table(det, sr, sk)
    plot_cookie_nsweep(det, sr, sk; outdir = outdir)
    return (det=det, sr=sr, sk=sk)
end

function print_nsweep_table(det, sr, sk)
    println("\n", repeat('=', 84))
    @printf("%-20s | %4s | %5s | %9s | %11s | %9s\n", "method", "n", "iters", "wall (s)", "true relres", "rank")
    println(repeat('-', 84))
    allm = vcat([("det", det)], [(l, sr[l]) for l in keys(sr)], [(l, sk[l]) for l in keys(sk)])
    for (lbl, dct) in allm, i in eachindex(dct[:n])
        @printf("%-20s | %4d | %5d | %9.1f | %11.2e | %9d\n", lbl, dct[:n][i], dct[:iters][i], dct[:time][i], dct[:relres][i],
                isempty(dct[:peak]) ? dct[:final][i] : dct[:peak][i])
    end
    println(repeat('=', 84))
end

function plot_cookie_nsweep(det, sr, sk; outdir)
    CairoMakie.activate!(type = "pdf"); mkpath(outdir)
    fig = Figure(size = (1800, 500))
    Label(fig[0, 1:3], "Cookie TT-GMRES — parameter-sample sweep (7 cookies, n1=1681)"; fontsize = 14, tellwidth = false)
    det_time = Dict(det[:n][i] => det[:time][i] for i in eachindex(det[:n]))

    ax1 = Axis(fig[1, 1], xlabel = "parameter samples n", ylabel = "wall time (s)",
               xscale = log10, yscale = log10, title = "wall time vs n (all solvers)", titlesize = 12)
    if !isempty(det[:n])
        c, m = variant_style("det")
        scatterlines!(ax1, det[:n], det[:time]; color = c, marker = m, markersize = 11, linewidth = 2, linestyle = :dash, label = "deterministic")
    end
    for l in keys(sr); d = sr[l]; isempty(d[:n]) && continue; c, m = variant_style(replace(l, " (S+R)"=>"")); scatterlines!(ax1, d[:n], d[:time]; color = c, marker = m, markersize = 9, linewidth = 2, label = l); end
    for l in keys(sk); d = sk[l]; isempty(d[:n]) && continue; scatterlines!(ax1, d[:n], d[:time]; color = :royalblue, marker = :circle, markersize = 9, linewidth = 2, label = l); end
    try; axislegend(ax1; position = :lt, framevisible = true, labelsize = 9); catch; end

    ax2 = Axis(fig[1, 2], xlabel = "parameter samples n", ylabel = "peak TT-rank",
               xscale = log10, title = "peak (S+R) / final (sketched) rank vs n", titlesize = 12)
    if !isempty(det[:n]); c, m = variant_style("det"); scatterlines!(ax2, det[:n], det[:peak]; color = c, marker = m, markersize = 11, linewidth = 2, linestyle = :dash, label = "det peak"); end
    for l in keys(sr); d = sr[l]; isempty(d[:n]) && continue; c, m = variant_style(replace(l, " (S+R)"=>"")); scatterlines!(ax2, d[:n], d[:peak]; color = c, marker = m, markersize = 9, linewidth = 2, label = l); end
    for l in keys(sk); d = sk[l]; isempty(d[:n]) && continue; scatterlines!(ax2, d[:n], d[:final]; color = :royalblue, marker = :circle, markersize = 9, linewidth = 2, linestyle = :dot, label = l); end
    try; axislegend(ax2; position = :rt, framevisible = true, labelsize = 9); catch; end

    ax3 = Axis(fig[1, 3], xlabel = "parameter samples n", ylabel = "speedup (t_det(n) / t_method(n))",
               xscale = log10, yscale = log10, title = "per-n speedup vs deterministic", titlesize = 12)
    hlines!(ax3, [1.0]; color = :black, linestyle = :dash, linewidth = 1)
    for l in keys(sr)
        d = sr[l]; isempty(d[:n]) && continue
        ns_ = [nn for nn in d[:n] if haskey(det_time, nn)]; isempty(ns_) && continue
        sp = [det_time[nn] / d[:time][findfirst(==(nn), d[:n])] for nn in ns_]
        c, m = variant_style(replace(l, " (S+R)"=>"")); scatterlines!(ax3, ns_, sp; color = c, marker = m, markersize = 9, linewidth = 2, label = l)
    end
    for l in keys(sk)
        d = sk[l]; isempty(d[:n]) && continue
        ns_ = [nn for nn in d[:n] if haskey(det_time, nn)]; isempty(ns_) && continue
        sp = [det_time[nn] / d[:time][findfirst(==(nn), d[:n])] for nn in ns_]
        scatterlines!(ax3, ns_, sp; color = :royalblue, marker = :circle, markersize = 9, linewidth = 2, label = l)
    end
    try; axislegend(ax3; position = :lt, framevisible = true, labelsize = 9); catch; end

    fname = joinpath(outdir, "cookie_nsweep.pdf"); save(fname, fig)
    println("\n→ saved $fname"); return fig
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_cookie_nsweep()
end
