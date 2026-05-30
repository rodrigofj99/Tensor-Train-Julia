using TensorTrains
using LinearAlgebra
using Random
using Statistics
using Printf
using CairoMakie
using LaTeXStrings

"""
Benchmark for `ttrand_rounding_adaptive` on the same `x0 + ε·noise` test family as
`examples/randomized_rounding_comparison.jl`. Fixed-rank methods (det/ttrand) round to
`base_rank`; adaptive sweeps a *tolerance factor* `ε_target = tol_factor · ε_pert`.

Reference object is the un-normalised linear combination `x0 + ε·noise` (which is what
all randomized methods operate on via the `[1, ε], [x0, noise]` linear-combo form), so
the err measures `‖method_output - target‖ / ‖target‖`.
"""

function setup(N::Int, d::Int, base_rank::Int, n_summands::Int,
               perturbation_rank::Int, seed::Int)
    @assert base_rank % n_summands == 0
    dims = ntuple(i -> d, N)
    Random.seed!(seed)
    summand_rank = base_rank ÷ n_summands
    summands = [rand_tt(Float64, dims, summand_rank;
                        orthogonal=false, normalise=true, stable=false)
                for _ = 1:n_summands]
    x0 = sum(summands); x0 = x0 / norm(x0)
    noise = rand_tt(Float64, dims, perturbation_rank;
                    orthogonal=false, normalise=true, stable=false)
    noise = noise / norm(noise)
    return x0, noise
end

"""
Run one (N, d, base_rank, n_summands) configuration. Returns a Dict with the data
used for plotting.
"""
function run_config(; N::Int, d::Int, base_rank::Int, n_summands::Int,
                      perturbation_rank::Int = 10,
                      εs = 10.0 .^ (-1:-1:-6),
                      tol_factors = (2.0, 10.0, 100.0),
                      n_realisations::Int = 5,
                      block_rks_adapt::Int = N,   # match the analysis (TT-Stack isometry)
                      seed::Int = 1234,
                      verbose_header::Bool = true)

    x0, noise = setup(N, d, base_rank, n_summands, perturbation_rank, seed)
    if verbose_header
        println("\n" * "="^120)
        println("Config: N=$N, d=$d, base_rank=$base_rank, n_summands=$n_summands, perturbation_rank=$perturbation_rank, block_rks=$block_rks_adapt, realisations=$n_realisations")
        println("="^120)
    end

    factor_headers = join((@sprintf("adapt(τ=%-3.0fε)", f) for f in tol_factors), " │ ")
    @printf("%-8s │ %-10s │ %-10s │ %s\n", "ε_pert", "det@$base_rank", "ttrand@$base_rank", factor_headers)
    println("─"^120)

    # Storage for plotting
    εs_vec   = collect(εs)
    err_det  = Float64[]
    err_rand = Float64[]
    err_rand_q25 = Float64[]; err_rand_q75 = Float64[]
    err_ad   = [Float64[] for _ in tol_factors]
    err_ad_q25 = [Float64[] for _ in tol_factors]
    err_ad_q75 = [Float64[] for _ in tol_factors]
    rk_ad    = [Int[]      for _ in tol_factors]

    for ε in εs
        target = x0 + ε*noise
        target_norm = norm(target)

        x_det = tt_rounding(target; tol=1e-16, rmax=base_rank)
        push!(err_det, norm(x_det - target) / target_norm)

        errs_rand = Float64[]
        for r = 1:n_realisations
            x_rand = ttrand_rounding([1.0, ε], [x0, noise], base_rank;
                                      orthogonal=true, block_rks=block_rks_adapt,
                                      seed=seed + 1000*r)
            push!(errs_rand, norm(x_rand - target) / target_norm)
        end
        push!(err_rand, median(errs_rand))
        push!(err_rand_q25, quantile(errs_rand, 0.25))
        push!(err_rand_q75, quantile(errs_rand, 0.75))

        adapt_cols = String[]
        for (j, tf) in pairs(tol_factors)
            ε_target = tf * ε
            errs_ad = Float64[]; ranks_ad = Int[]
            for r = 1:n_realisations
                x_ad = ttrand_rounding_adaptive([1.0, ε], [x0, noise], ε_target;
                                                 block_rks=block_rks_adapt,
                                                 seed=seed + 1000*r)
                push!(errs_ad, norm(x_ad - target) / target_norm)
                push!(ranks_ad, maximum(x_ad.ttv_rks))
            end
            push!(err_ad[j],     median(errs_ad))
            push!(err_ad_q25[j], quantile(errs_ad, 0.25))
            push!(err_ad_q75[j], quantile(errs_ad, 0.75))
            push!(rk_ad[j],      round(Int, median(ranks_ad)))
            push!(adapt_cols, @sprintf("rk%-3d err %.1e", round(Int, median(ranks_ad)), median(errs_ad)))
        end
        @printf("%-8.0e │ %-10.2e │ %-10.2e │ %s\n", ε, err_det[end], err_rand[end], join(adapt_cols, " │ "))
    end

    return (N=N, d=d, base_rank=base_rank, n_summands=n_summands,
            perturbation_rank=perturbation_rank, block_rks=block_rks_adapt,
            tol_factors=collect(tol_factors),
            εs=εs_vec,
            err_det=err_det,
            err_rand=err_rand, err_rand_q25=err_rand_q25, err_rand_q75=err_rand_q75,
            err_ad=err_ad, err_ad_q25=err_ad_q25, err_ad_q75=err_ad_q75,
            rk_ad=rk_ad)
end

"""
Plot error and max rank vs ε_pert for all methods of one config. Saves a PDF with
two side-by-side panels: relative error (log/log) on the left, max output rank
(linear y) on the right.
"""
function plot_config(res; dir = "out/randomized_rounding_adaptive")
    mkpath(dir)
    CairoMakie.activate!(type = "pdf")

    title_str = "N=$(res.N), d=$(res.d), base_rank=$(res.base_rank), n_summands=$(res.n_summands), block_rks=$(res.block_rks)"
    fig = Figure(size = (1200, 460))
    Label(fig[0, 1:2], title_str; fontsize=13, tellwidth=false)

    ax_err = Axis(fig[1, 1],
                  xlabel = L"\varepsilon_\mathrm{pert}",
                  ylabel = L"\Vert \hat{x} - x_\varepsilon \Vert / \Vert x_\varepsilon \Vert",
                  xscale = log10, yscale = log10, xreversed = true,
                  title  = "Relative error",
                  titlesize = 12)
    ax_rk  = Axis(fig[1, 2],
                  xlabel = L"\varepsilon_\mathrm{pert}",
                  ylabel = "max output rank",
                  xscale = log10, xreversed = true,
                  title  = "Output max rank",
                  titlesize = 12)

    palette = [:tomato, :royalblue, :seagreen, :orchid, :goldenrod]
    markers = [:utriangle, :rect, :star5, :hexagon, :dtriangle]

    # --- ERROR PANEL ---
    scatterlines!(ax_err, res.εs, max.(res.err_det, 1e-18);
                  color=:black, linewidth=2, marker=:circle, markersize=8,
                  label="det @ rk=$(res.base_rank)")
    band!(ax_err, res.εs, max.(res.err_rand_q25, 1e-18), max.(res.err_rand_q75, 1e-18);
          color=(:gray, 0.25))
    scatterlines!(ax_err, res.εs, max.(res.err_rand, 1e-18);
                  color=:gray, linewidth=2, marker=:diamond, markersize=8,
                  linestyle=:dash, label="ttrand @ rk=$(res.base_rank)")
    for (j, tf) in pairs(res.tol_factors)
        c = palette[mod1(j, length(palette))]
        m = markers[mod1(j, length(markers))]
        band!(ax_err, res.εs, max.(res.err_ad_q25[j], 1e-18), max.(res.err_ad_q75[j], 1e-18);
              color=(c, 0.20))
        scatterlines!(ax_err, res.εs, max.(res.err_ad[j], 1e-18);
                      color=c, linewidth=2, marker=m, markersize=8,
                      label=L"\text{adapt}(\tau = %$(Int(tf))\,\varepsilon)")
    end
    lines!(ax_err, res.εs, res.εs; color=:black, linestyle=:dot, linewidth=1,
           label=L"\varepsilon")
    axislegend(ax_err; position=:rb, framevisible=true, labelsize=10)

    # --- RANK PANEL ---
    # Fixed-rank methods sit at base_rank (drawn as horizontal references).
    base_rks = fill(res.base_rank, length(res.εs))
    scatterlines!(ax_rk, res.εs, base_rks;
                  color=:black, linewidth=2, marker=:circle, markersize=8,
                  label="det / ttrand @ rk=$(res.base_rank)")

    for (j, tf) in pairs(res.tol_factors)
        c = palette[mod1(j, length(palette))]
        m = markers[mod1(j, length(markers))]
        scatterlines!(ax_rk, res.εs, res.rk_ad[j];
                      color=c, linewidth=2, marker=m, markersize=8,
                      label=L"\text{adapt}(\tau = %$(Int(tf))\,\varepsilon)")
    end
    axislegend(ax_rk; position=:rt, framevisible=true, labelsize=10)

    fname = "$(dir)/N$(res.N)_d$(res.d)_rk$(res.base_rank)_summands$(res.n_summands)_blk$(res.block_rks).pdf"
    save(fname, fig)
    println("  → saved $fname")
    return fig
end

# === configurations ===
configs = [
    (N=10, d=4, base_rank=8,  n_summands=8,  perturbation_rank=10),
    (N=10, d=4, base_rank=8,  n_summands=1,  perturbation_rank=10),
    (N=20, d=4, base_rank=8,  n_summands=8,  perturbation_rank=10),
    (N=20, d=4, base_rank=16, n_summands=16, perturbation_rank=10),
    (N=20, d=4, base_rank=16, n_summands=1,  perturbation_rank=10),
]

results = [run_config(; cfg...) for cfg in configs]

println("\n--- plotting ---")
for res in results
    plot_config(res)
end
