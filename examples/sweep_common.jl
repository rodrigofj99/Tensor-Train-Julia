using CairoMakie
using Statistics
using Serialization
using Printf

"""
Shared scaffolding for the block-rank sweep experiments (Studies M / S and the cookie studies).
Factor the median/quantile aggregation, the incremental `.jls` dumping, the per-method plot style
registry, and the two reusable figure layouts (the 4-panel tolerance sweep and the paper Fig-5
error/speedup/compression panel) that were previously copy-pasted across the sweep scripts.

`include("sweep_common.jl")` — a plain file (repo convention), not a package module.
"""

# Median + interquartile spread of a trial vector.
trial_stats(v) = (med = median(v), q25 = quantile(v, 0.25), q75 = quantile(v, 0.75))

# mkpath + serialize (the `partial_*.jls` / aggregate-dump pattern).
function dump_incremental(dir::AbstractString, name::AbstractString, payload)
    mkpath(dir); serialize(joinpath(dir, name), payload); return nothing
end

# Canonical (color, marker) per method label so every figure is visually consistent. Unknown labels
# fall back to a neutral grey circle.
const VARIANT_STYLE = Dict{String,Tuple{Symbol,Symbol}}(
    "strict-KRP" => (:purple,     :xcross),
    "KRP/Gauss"  => (:purple,     :xcross),
    "orth-KRP"   => (:tomato,     :circle),
    "KRP"        => (:tomato,     :circle),
    "TTStack-N"  => (:gold,       :star5),
    "TTStack-2N" => (:darkorange, :diamond),
    "TTStack-16" => (:darkorange, :diamond),
    "TTStack-32" => (:firebrick,  :rect),
    "Adap-R"     => (:seagreen,   :utriangle),
    "det"        => (:black,      :hexagon),
    "det (ref)"  => (:black,      :hexagon),
)
variant_style(lbl) = get(VARIANT_STYLE, lbl, (:gray, :circle))

"""
    four_panel_sweep(adapt, det; variant_labels, N, n_trials, title, dir, fname, ε_profile=1e-6)

The tolerance-sweep figure used by the single-TT studies. `adapt[label]` and `det` are the Dicts the
sweep drivers fill (keys `:ε, :rk_med, :rk_q25, :rk_q75, :err_med, :err_q25, :err_q75, :time_med,
:time_q25, :time_q75, :rk_profile, :rk_profile_q25, :rk_profile_q75` for adapt; `:ε, :rk, :err,
:time_med, :rk_profile` for det). Panels: max rank vs ε; achieved error vs ε; per-bond rank profile
at `ε_profile`; wall time vs achieved error.
"""
function four_panel_sweep(adapt::AbstractDict, det::AbstractDict; variant_labels, N, n_trials,
                          title::AbstractString, dir::AbstractString, fname::AbstractString,
                          ε_profile::Float64 = 1e-6)
    CairoMakie.activate!(type = "pdf"); mkpath(dir)
    fig = Figure(size = (1400, 900))
    Label(fig[0, 1:2], title; fontsize = 13, tellwidth = false)

    ax_rk = Axis(fig[1, 1], xlabel = L"\varepsilon", ylabel = "max output rank",
                 xscale = log10, xreversed = true, title = "max output rank vs tolerance", titlesize = 12)
    for lbl in variant_labels
        d = adapt[lbl]; c, m = variant_style(lbl)
        band!(ax_rk, d[:ε], d[:rk_q25], d[:rk_q75]; color = (c, 0.25))
        scatterlines!(ax_rk, d[:ε], d[:rk_med]; color = c, linewidth = 2, marker = m, markersize = 9, label = lbl)
    end
    c, m = variant_style("det")
    scatterlines!(ax_rk, det[:ε], det[:rk]; color = c, linewidth = 2, marker = m, markersize = 10, label = "det @ tol")
    axislegend(ax_rk; position = :lt, framevisible = true, labelsize = 10)

    ax_err = Axis(fig[1, 2], xlabel = L"\varepsilon",
                  ylabel = L"\Vert \hat{x} - x_\mathrm{ref}\Vert/\Vert x_\mathrm{ref}\Vert",
                  xscale = log10, yscale = log10, xreversed = true,
                  title = "achieved error vs tolerance", titlesize = 12)
    εs = sort(collect(adapt[first(variant_labels)][:ε]))
    lines!(ax_err, εs, εs; color = :gray, linewidth = 1, linestyle = :dot, label = L"\text{err}=\varepsilon")
    for lbl in variant_labels
        d = adapt[lbl]; c, m = variant_style(lbl)
        band!(ax_err, d[:ε], max.(d[:err_q25], 1e-18), max.(d[:err_q75], 1e-18); color = (c, 0.25))
        scatterlines!(ax_err, d[:ε], max.(d[:err_med], 1e-18); color = c, linewidth = 2, marker = m, markersize = 9, label = lbl)
    end
    c, m = variant_style("det")
    scatterlines!(ax_err, det[:ε], max.(det[:err], 1e-18); color = c, linewidth = 2, marker = m, markersize = 10, label = "det @ tol")
    axislegend(ax_err; position = :lt, framevisible = true, labelsize = 10)

    pr = findfirst(ε -> isapprox(ε, ε_profile; rtol = 1e-12), adapt[first(variant_labels)][:ε])
    ax_pr = Axis(fig[2, 1], xlabel = "bond index k", ylabel = "TT rank at bond k", yscale = log10,
                 title = (@sprintf("per-bond rank profile at ε=%.0e", ε_profile)), titlesize = 12)
    if pr !== nothing
        for lbl in variant_labels
            d = adapt[lbl]; c, _ = variant_style(lbl)
            prof = d[:rk_profile][pr]; q25 = d[:rk_profile_q25][pr]; q75 = d[:rk_profile_q75][pr]
            band!(ax_pr, 1:length(prof), max.(q25, 1), max.(q75, 1); color = (c, 0.25))
            lines!(ax_pr, 1:length(prof), max.(prof, 1); color = c, linewidth = 2, label = lbl)
        end
        dp = det[:rk_profile][pr]
        lines!(ax_pr, 1:length(dp), max.(dp, 1); color = :black, linewidth = 2, linestyle = :dash, label = "det @ tol")
        axislegend(ax_pr; position = :rt, framevisible = true, labelsize = 10)
    end

    ax_t = Axis(fig[2, 2], xlabel = L"\Vert \hat{x} - x_\mathrm{ref}\Vert/\Vert x_\mathrm{ref}\Vert",
                ylabel = "wall time (s)", xscale = log10, yscale = log10, xreversed = true,
                title = "wall time vs achieved error", titlesize = 12)
    for lbl in variant_labels
        d = adapt[lbl]; c, m = variant_style(lbl)
        band!(ax_t, max.(d[:err_med], 1e-18), max.(d[:time_q25], 1e-4), max.(d[:time_q75], 1e-4); color = (c, 0.25))
        scatterlines!(ax_t, max.(d[:err_med], 1e-18), max.(d[:time_med], 1e-4); color = c, linewidth = 2, marker = m, markersize = 9, label = lbl)
    end
    c, m = variant_style("det")
    scatterlines!(ax_t, max.(det[:err], 1e-18), max.(det[:time_med], 1e-4); color = c, linewidth = 2, marker = m, markersize = 10, label = "det @ tol")
    axislegend(ax_t; position = :lt, framevisible = true, labelsize = 10)

    f = joinpath(dir, fname); save(f, fig); println("→ saved $f"); return fig
end

"""
    speedup_compression_panel(adapt, det; variant_labels, dims, title, dir, fname)

The paper Fig-5 layout: relative error, speedup (= t_det / t_rand), and **compression** vs tolerance
ε. Speedup is per-ε against the deterministic baseline (`det[:time_med]`). Compression is the achieved
*storage ratio* of each method's rounded TT — `Σₖ rₖ₋₁·nₖ·rₖ / Πₖ nₖ` (TT core storage divided by the
number of entries of the full tensor) — computed from the per-bond rank profile (`:rk_profile`) and
the mode dimensions `dims` (length N). Lower is better (more compressed).
"""
function speedup_compression_panel(adapt::AbstractDict, det::AbstractDict; variant_labels, dims,
                                   title::AbstractString, dir::AbstractString, fname::AbstractString)
    CairoMakie.activate!(type = "pdf"); mkpath(dir)
    det_time = Dict(det[:ε][i] => det[:time_med][i] for i in eachindex(det[:ε]))
    full_entries = prod(Float64.(collect(dims)))
    _storage(prof) = sum(prof[k]*dims[k]*prof[k+1] for k in 1:length(dims))   # Σ rₖ₋₁·nₖ·rₖ
    _comp(profiles) = [_storage(p)/full_entries for p in profiles]
    fig = Figure(size = (1500, 460))
    Label(fig[0, 1:3], title; fontsize = 13, tellwidth = false)

    ax_e = Axis(fig[1, 1], xlabel = L"\varepsilon", ylabel = "relative error",
                xscale = log10, yscale = log10, xreversed = true, title = "accuracy", titlesize = 12)
    εs = sort(collect(adapt[first(variant_labels)][:ε]))
    lines!(ax_e, εs, εs; color = :gray, linewidth = 1, linestyle = :dot, label = L"\text{err}=\varepsilon")
    for lbl in variant_labels
        d = adapt[lbl]; c, m = variant_style(lbl)
        scatterlines!(ax_e, d[:ε], max.(d[:err_med], 1e-18); color = c, linewidth = 2, marker = m, markersize = 9, label = lbl)
    end
    axislegend(ax_e; position = :lt, framevisible = true, labelsize = 9)

    ax_s = Axis(fig[1, 2], xlabel = L"\varepsilon", ylabel = "speedup (t_det / t_rand)",
                xscale = log10, xreversed = true, title = "speedup vs deterministic", titlesize = 12)
    hlines!(ax_s, [1.0]; color = :black, linestyle = :dash, linewidth = 1)
    for lbl in variant_labels
        d = adapt[lbl]; c, m = variant_style(lbl)
        sp = [det_time[ε] / t for (ε, t) in zip(d[:ε], d[:time_med])]
        scatterlines!(ax_s, d[:ε], sp; color = c, linewidth = 2, marker = m, markersize = 9, label = lbl)
    end
    axislegend(ax_s; position = :lt, framevisible = true, labelsize = 9)

    ax_c = Axis(fig[1, 3], xlabel = L"\varepsilon", ylabel = "TT storage / full-tensor entries",
                xscale = log10, yscale = log10, xreversed = true,
                title = "compression (storage fraction, lower = better)", titlesize = 12)
    for lbl in variant_labels
        d = adapt[lbl]; c, m = variant_style(lbl)
        scatterlines!(ax_c, d[:ε], _comp(d[:rk_profile]); color = c, linewidth = 2, marker = m, markersize = 9, label = lbl)
    end
    cd, md = variant_style("det")
    scatterlines!(ax_c, det[:ε], _comp(det[:rk_profile]); color = cd, linewidth = 2, marker = md, markersize = 10, label = "det @ tol")
    axislegend(ax_c; position = :lt, framevisible = true, labelsize = 9)

    f = joinpath(dir, fname); save(f, fig); println("→ saved $f"); return fig
end
