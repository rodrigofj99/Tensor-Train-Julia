using TensorTrains
using LinearAlgebra
using Random
using Statistics
using Printf
using CairoMakie
using QuanticsTCI

include("quantics_example.jl")

"""
Compare five adaptive variants vs deterministic `tt_rounding` on the same
quantics 3-way Hadamard product, sweeping tolerance 1e-1..1e-8.

`block_rks` is the per-block rank of the *initial* TTStack sketch (driving
`y_norm` accuracy); `block_rks_inc` is the per-block rank of *extension*
sketches built inside the inner loop when the sliding window of sketch
cols is exhausted — abbreviated `ext` in the variant labels. Neither
controls Q's per-iteration rank growth (which is geometric `0.2·rk`
via `ℓ_inc`).

  - `KRP`       : block_rks=1  (initial) + block_rks_inc=1 (Khatri-Rao baseline)
  - `N/ext=1`   : block_rks=N + block_rks_inc=1
  - `N/ext=8`   : block_rks=N + block_rks_inc=8
  - `N/ext=16`  : block_rks=N + block_rks_inc=16 (≈ N÷4 for N=60, current default)
  - `N/ext=32`  : block_rks=N + block_rks_inc=32 (≈ N÷2)

The deterministic baseline includes the cost of forming the Kronecker
`raw_product = tts[1]*tts[2]*tts[3]` on each trial (the cost a user pays
when starting from the NTuple input — the randomized algorithms never form it).
"""

function generate_quantics_tensors(R::Int; tolerance=1e-10, seed=1234)
    Random.seed!(seed)
    println("Generating QuanticsTCI tensor trains (R=$R, tol=$tolerance)…")
    xvals = range(0.0, 1.0; length=2^R)
    fs = ((x,y,z) -> term1(x,y,z,R), (x,y,z) -> term2(x,y,z,R), (x,y,z) -> term3(x,y,z,R))
    interpolants = ntuple(j -> quanticscrossinterpolate(Float64, fs[j],
                                                        [xvals, xvals, xvals];
                                                        tolerance=tolerance,
                                                        unfoldingscheme=:interleaved), 3)
    qtt = getindex.(interpolants, 1)
    ttv = qtt_to_ttvector.(qtt)
    for i = 1:3
        println("  term $i: max rank = $(maximum(ttv[i].ttv_rks))")
    end
    return tuple(ttv...)
end

function run_blockrks_sweep(; R::Int = 20,
                              tci_tol::Float64 = 1e-10,
                              ref_tol::Float64 = 1e-10,
                              εs = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8),
                              ℓ_min::Int = 4,
                              ℓ_inc::Int = 4,
                              n_trials::Int = 1000,
                              seed::Int = 1234,
                              dir::String = "out/quantics_hadamard_blockrks_sweep")
    mkpath(dir)
    tts = generate_quantics_tensors(R; tolerance=tci_tol, seed=seed)
    N = tts[1].N
    raw_product = tts[1] * tts[2] * tts[3]
    println("Raw product max rank = $(maximum(raw_product.ttv_rks))")
    reference = tt_rounding(raw_product; tol=ref_tol)
    println("Reference (det @ tol=$(ref_tol)) max rank = $(maximum(reference.ttv_rks))")
    ref_norm  = norm(reference)
    println("Reference norm = $(round(ref_norm, sigdigits=6))")

    # Four variants — fix init at N for the last three, sweep inc; first is pure KRP.
    variants = (
      (label="KRP",      blk=1, inc=1),
      (label="N/ext=1",  blk=N, inc=1),
      (label="N/ext=8",  blk=N, inc=8),
      (label="N/ext=16", blk=N, inc=16),
      (label="N/ext=32", blk=N, inc=32),
    )

    # JIT warmup (untimed)
    println("\n--- Warmup ---")
    for v in variants
        _ = ttrand_rounding_adaptive(tts, first(εs); ℓ_min=ℓ_min, ℓ_inc=ℓ_inc,
                                      block_rks=v.blk, block_rks_inc=v.inc, seed=seed)
    end
    _ = tt_rounding(raw_product; tol=first(εs))

    # --- Adaptive sweeps per variant ---
    adapt_results = Dict{String, Dict{Symbol, Any}}()
    for v in variants
        println("\n--- Adaptive sweep $(v.label) (block_rks=$(v.blk), block_rks_inc=$(v.inc), ℓ_inc=$ℓ_inc) ---")
        d = Dict{Symbol, Any}(:ε => Float64[], :rk_med => Float64[],
                              :err_med => Float64[], :err_q25 => Float64[],
                              :err_q75 => Float64[],
                              :rk_q25 => Float64[], :rk_q75 => Float64[],
                              :time_med => Float64[], :time_q25 => Float64[],
                              :time_q75 => Float64[],
                              :rk_profile => Vector{Vector{Int}}(),
                              :rk_profile_q25 => Vector{Vector{Float64}}(),
                              :rk_profile_q75 => Vector{Vector{Float64}}())
        for ε in εs
            errs = Float64[]; rks = Float64[]; times = Float64[]; rk_profiles = Vector{Int}[]
            for t = 1:n_trials
                local ŷ
                tm = @elapsed ŷ = ttrand_rounding_adaptive(tts, ε;
                                                           ℓ_min=ℓ_min, ℓ_inc=ℓ_inc,
                                                           block_rks=v.blk,
                                                           block_rks_inc=v.inc,
                                                           seed=seed + 1000*t)
                push!(errs, norm(reference - ŷ) / ref_norm)
                push!(rks, Float64(maximum(ŷ.ttv_rks)))
                push!(times, tm)
                push!(rk_profiles, ŷ.ttv_rks)
            end
            push!(d[:ε], ε)
            push!(d[:err_med], median(errs))
            push!(d[:err_q25], quantile(errs, 0.25))
            push!(d[:err_q75], quantile(errs, 0.75))
            push!(d[:rk_med],  median(rks))
            push!(d[:rk_q25],  quantile(rks, 0.25))
            push!(d[:rk_q75],  quantile(rks, 0.75))
            push!(d[:time_med], median(times))
            push!(d[:time_q25], quantile(times, 0.25))
            push!(d[:time_q75], quantile(times, 0.75))
            n_bonds_p1 = length(rk_profiles[1])
            med_profile = [round(Int, median([p[k] for p in rk_profiles])) for k = 1:n_bonds_p1]
            q25_profile = [quantile([Float64(p[k]) for p in rk_profiles], 0.25) for k = 1:n_bonds_p1]
            q75_profile = [quantile([Float64(p[k]) for p in rk_profiles], 0.75) for k = 1:n_bonds_p1]
            push!(d[:rk_profile], med_profile)
            push!(d[:rk_profile_q25], q25_profile)
            push!(d[:rk_profile_q75], q75_profile)
            @printf("  ε=%.0e → max rk=%.0f, err=%.3e, t=%.3fs (range %.2e–%.2e)\n",
                    ε, median(rks), median(errs), median(times), minimum(errs), maximum(errs))
        end
        adapt_results[v.label] = d
    end

    # --- Deterministic baseline at matched tolerance ---
    # Form the Kronecker raw_product inside the timer per trial — this is the
    # cost a user pays when starting from the NTuple input.
    println("\n--- Deterministic baseline (matched tol, raw_product formation timed) ---")
    det = Dict{Symbol, Any}(:ε => Float64[], :rk => Float64[], :err => Float64[],
                            :time_med => Float64[], :time_q25 => Float64[],
                            :time_q75 => Float64[],
                            :rk_profile => Vector{Vector{Int}}())
    for ε in εs
        local ŷ
        times = Float64[]
        for _ in 1:n_trials
            push!(times, @elapsed ŷ = tt_rounding(tts[1] * tts[2] * tts[3]; tol=ε))
        end
        err = norm(reference - ŷ) / ref_norm
        push!(det[:ε], ε)
        push!(det[:rk], Float64(maximum(ŷ.ttv_rks)))
        push!(det[:err], err)
        push!(det[:time_med], median(times))
        push!(det[:time_q25], quantile(times, 0.25))
        push!(det[:time_q75], quantile(times, 0.75))
        push!(det[:rk_profile], ŷ.ttv_rks)
        @printf("  tol=%.0e → max rk=%d, err=%.3e, t=%.3fs\n",
                ε, maximum(ŷ.ttv_rks), err, median(times))
    end

    variant_labels = [v.label for v in variants]
    plot_blockrks_sweep(adapt_results, det; R=R, N=N, variant_labels=variant_labels,
                        n_trials=n_trials, ℓ_inc=ℓ_inc, dir=dir)
    return (adapt=adapt_results, det=det)
end

const _VARIANT_COLOURS = Dict("KRP"      => :tomato,
                              "N/ext=1"  => :gold,
                              "N/ext=8"  => :darkorange,
                              "N/ext=16" => :firebrick,
                              "N/ext=32" => :darkred)
const _VARIANT_MARKERS = Dict("KRP"      => :circle,
                              "N/ext=1"  => :star5,
                              "N/ext=8"  => :diamond,
                              "N/ext=16" => :utriangle,
                              "N/ext=32" => :rect)

function plot_blockrks_sweep(adapt, det; R, N, variant_labels, n_trials, ℓ_inc, dir)
    CairoMakie.activate!(type="pdf")
    title_str = "Block-rank-extension sweep (R=$R, N=$N, $(n_trials) trials, ℓ_inc=$ℓ_inc)"

    fig = Figure(size=(1400, 900))
    Label(fig[0, 1:2], title_str; fontsize=13, tellwidth=false)

    # --- Panel 1: max rank vs tolerance ---
    ax_rk = Axis(fig[1, 1],
                 xlabel = L"\varepsilon \;(\text{tolerance})",
                 ylabel = "max output rank",
                 xscale = log10, xreversed = true,
                 title = "max output rank vs tolerance",
                 titlesize = 12)
    for lbl in variant_labels
        d = adapt[lbl]
        c = _VARIANT_COLOURS[lbl]
        m = _VARIANT_MARKERS[lbl]
        band!(ax_rk, d[:ε], d[:rk_q25], d[:rk_q75]; color=(c, 0.25))
        scatterlines!(ax_rk, d[:ε], d[:rk_med]; color=c, linewidth=2,
                      marker=m, markersize=9, label="adapt $lbl")
    end
    scatterlines!(ax_rk, det[:ε], det[:rk]; color=:black, linewidth=2,
                  marker=:hexagon, markersize=10, label="det @ tol")
    axislegend(ax_rk; position=:lt, framevisible=true, labelsize=10)

    # --- Panel 2: achieved err vs tolerance, with err = ε reference ---
    ax_err = Axis(fig[1, 2],
                  xlabel = L"\varepsilon \;(\text{tolerance})",
                  ylabel = L"\Vert \hat{x} - x_\mathrm{ref} \Vert / \Vert x_\mathrm{ref} \Vert",
                  xscale = log10, yscale = log10, xreversed = true,
                  title = "achieved error vs tolerance",
                  titlesize = 12)
    εs_sorted = sort(collect(adapt[first(variant_labels)][:ε]))
    lines!(ax_err, εs_sorted, εs_sorted; color=:gray, linewidth=1,
           linestyle=:dot, label=L"\text{err}=\varepsilon")
    for lbl in variant_labels
        d = adapt[lbl]
        c = _VARIANT_COLOURS[lbl]
        m = _VARIANT_MARKERS[lbl]
        band!(ax_err, d[:ε], max.(d[:err_q25], 1e-18), max.(d[:err_q75], 1e-18);
              color=(c, 0.25))
        scatterlines!(ax_err, d[:ε], max.(d[:err_med], 1e-18); color=c, linewidth=2,
                      marker=m, markersize=9, label="adapt $lbl")
    end
    scatterlines!(ax_err, det[:ε], max.(det[:err], 1e-18); color=:black, linewidth=2,
                  marker=:hexagon, markersize=10, label="det @ tol")
    axislegend(ax_err; position=:lt, framevisible=true, labelsize=10)

    # --- Panel 3: per-bond rank profile at ε=1e-6 ---
    ε_target = 1e-6
    pr_idx = findfirst(ε -> isapprox(ε, ε_target; rtol=1e-12), adapt[first(variant_labels)][:ε])
    ax_pr = Axis(fig[2, 1],
                 xlabel = "bond index k",
                 ylabel = "TT rank at bond k",
                 yscale = log10,
                 title = @sprintf("per-bond rank profile at ε=%.0e", ε_target),
                 titlesize = 12)
    for lbl in variant_labels
        d = adapt[lbl]
        c = _VARIANT_COLOURS[lbl]
        prof = d[:rk_profile][pr_idx]
        q25  = d[:rk_profile_q25][pr_idx]
        q75  = d[:rk_profile_q75][pr_idx]
        band!(ax_pr, 1:length(prof), max.(q25, 1), max.(q75, 1); color=(c, 0.25))
        lines!(ax_pr, 1:length(prof), max.(prof, 1);
               color=c, linewidth=2, label="adapt $lbl")
    end
    det_prof = det[:rk_profile][pr_idx]
    lines!(ax_pr, 1:length(det_prof), max.(det_prof, 1);
           color=:black, linewidth=2, linestyle=:dash, label="det @ tol")
    axislegend(ax_pr; position=:rt, framevisible=true, labelsize=10)

    # --- Panel 4: wall time vs achieved error ---
    ax_t = Axis(fig[2, 2],
                xlabel = L"\Vert \hat{x} - x_\mathrm{ref} \Vert / \Vert x_\mathrm{ref} \Vert",
                ylabel = "wall time (s)",
                xscale = log10, yscale = log10, xreversed = true,
                title = "wall time vs achieved error",
                titlesize = 12)
    for lbl in variant_labels
        d = adapt[lbl]
        c = _VARIANT_COLOURS[lbl]
        m = _VARIANT_MARKERS[lbl]
        band!(ax_t, max.(d[:err_med], 1e-18),
              max.(d[:time_q25], 1e-4), max.(d[:time_q75], 1e-4);
              color=(c, 0.25))
        scatterlines!(ax_t, max.(d[:err_med], 1e-18), max.(d[:time_med], 1e-4);
                      color=c, linewidth=2, marker=m, markersize=9, label="adapt $lbl")
    end
    band!(ax_t, max.(det[:err], 1e-18),
          max.(det[:time_q25], 1e-4), max.(det[:time_q75], 1e-4);
          color=(:black, 0.20))
    scatterlines!(ax_t, max.(det[:err], 1e-18), max.(det[:time_med], 1e-4);
                  color=:black, linewidth=2, marker=:hexagon, markersize=10,
                  label="det @ tol")
    axislegend(ax_t; position=:lt, framevisible=true, labelsize=10)

    fname = "$(dir)/quantics_hadamard_blockrks_ext_sweep_R$(R).pdf"
    save(fname, fig)
    println("\n→ saved $fname")
    return fig
end

run_blockrks_sweep()
