using TensorTrains
using LinearAlgebra
using Random
using Statistics
using Printf
using CairoMakie
using QuanticsTCI

include("quantics_example.jl")

"""
Adaptive companion to `quantics_hadamard_sketch_benchmark.jl`.

Uses the same QuanticsTCI 3-term Hadamard product (term1·term2·term3) as the input,
but sweeps the *tolerance* for `ttrand_rounding_adaptive` rather than the target rank
for fixed-rank methods. Reports the achieved (max rank, relative error) for each
tolerance, alongside the deterministic and fixed-rank-`ttrand_rounding` reference
curves at a sequence of target ranks. All methods are plotted on a common
(rank, err) plane.
"""

function generate_quantics_tensors(R::Int; tolerance=1e-10, seed=1234)
    Random.seed!(seed)
    println("Generating QuanticsTCI tensor trains (R=$R, tol=$tolerance)…")
    xvals = range(0.0, 1.0; length=2^R)
    yvals = range(0.0, 1.0; length=2^R)
    zvals = range(0.0, 1.0; length=2^R)
    f1(x,y,z) = term1(x,y,z,R); f2(x,y,z) = term2(x,y,z,R); f3(x,y,z) = term3(x,y,z,R)
    fs = (f1, f2, f3)

    interpolants = ntuple(j -> quanticscrossinterpolate(Float64, fs[j],
                                                         [xvals, yvals, zvals];
                                                         tolerance=tolerance,
                                                         unfoldingscheme=:interleaved), 3)
    qtt = getindex.(interpolants, 1)
    ttv = qtt_to_ttvector.(qtt)
    for i = 1:3
        println("  term $i: max rank = $(maximum(ttv[i].ttv_rks))")
    end
    return tuple(ttv...)
end

"""
Sweep adaptive tolerance and fixed-rank baselines on the same Hadamard product.
"""
function run_hadamard_adaptive_benchmark(; R::Int = 20,
                                           tci_tol::Float64 = 1e-10,
                                           ref_tol::Float64 = 1e-10,
                                           εs = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8),
                                           target_ranks = [8, 16, 32, 48, 64, 96, 128, 160, 192],
                                           block_rks_fixed::Int = 8,
                                           ℓ_min::Int = 4,
                                           ℓ_inc::Int = 8,
                                           n_trials::Int = 100,
                                           seed::Int = 1234,
                                           dir::String = "out/quantics_hadamard_adaptive")
    mkpath(dir)
    tts = generate_quantics_tensors(R; tolerance=tci_tol, seed=seed)
    N = tts[1].N
    raw_product = tts[1] * tts[2] * tts[3]
    println("Raw product max rank = $(maximum(raw_product.ttv_rks))")
    reference = tt_rounding(raw_product; tol=ref_tol)
    println("Reference (det @ tol=$(ref_tol)) max rank = $(maximum(reference.ttv_rks))")
    ref_norm  = norm(reference)
    println("Reference norm = $(round(ref_norm, sigdigits=6))")

    # --- Adaptive sweep over tolerances ---
    # Defaults: block_rks=N (low-variance y_norm), block_rks_inc=N÷4 (cheap incremental).
    println("\n--- Adaptive sweep (ℓ_min=$ℓ_min, ℓ_inc=$ℓ_inc, block_rks=N=$N / inc=N÷4=$(N÷4) orth) ---")
    # JIT warmup (untimed) — first call would otherwise pay compilation cost
    _ = ttrand_rounding_adaptive(tts, first(εs); ℓ_min=ℓ_min, ℓ_inc=ℓ_inc, seed=seed)
    adapt = Dict{Symbol, Any}(:ε => Float64[], :rk_med => Float64[],
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
                                                       seed=seed + 1000*t)
            push!(errs, norm(reference - ŷ) / ref_norm)
            push!(rks,  Float64(maximum(ŷ.ttv_rks)))
            push!(times, tm)
            push!(rk_profiles, ŷ.ttv_rks)
        end
        push!(adapt[:ε],      ε)
        push!(adapt[:err_med], median(errs))
        push!(adapt[:err_q25], quantile(errs, 0.25))
        push!(adapt[:err_q75], quantile(errs, 0.75))
        push!(adapt[:rk_med],  median(rks))
        push!(adapt[:rk_q25],  quantile(rks, 0.25))
        push!(adapt[:rk_q75],  quantile(rks, 0.75))
        push!(adapt[:time_med], median(times))
        push!(adapt[:time_q25], quantile(times, 0.25))
        push!(adapt[:time_q75], quantile(times, 0.75))

        n_bonds_p1 = length(rk_profiles[1])
        med_profile = [round(Int, median([p[k] for p in rk_profiles])) for k = 1:n_bonds_p1]
        q25_profile = [quantile([Float64(p[k]) for p in rk_profiles], 0.25) for k = 1:n_bonds_p1]
        q75_profile = [quantile([Float64(p[k]) for p in rk_profiles], 0.75) for k = 1:n_bonds_p1]
        push!(adapt[:rk_profile],     med_profile)
        push!(adapt[:rk_profile_q25], q25_profile)
        push!(adapt[:rk_profile_q75], q75_profile)
        @printf("  ε=%.0e → max rk=%.0f, err=%.3e, t=%.2fs (range %.2e–%.2e)\n",
                ε, median(rks), median(errs), median(times), minimum(errs), maximum(errs))
        println("           median rk profile: ", med_profile)
    end

    # --- Deterministic baseline: tt_rounding of the exact product at each target_rank ---
    println("\n--- Deterministic baseline (target rk) ---")
    _ = tt_rounding(raw_product; rmax=first(target_ranks))  # JIT warmup
    det = Dict{Symbol, Any}(:rk => Float64[], :err => Float64[],
                            :time_med => Float64[], :time_q25 => Float64[],
                            :time_q75 => Float64[],
                            :rk_profile => Vector{Vector{Int}}())
    # NOTE: each trial includes the cost of forming the Kronecker `raw_product`
    # from `tts`, since that's the cost a user pays when they only have the
    # NTuple input (the randomized algorithms never form it).
    for tr in target_ranks
        local ŷ
        times = Float64[]
        for _ in 1:n_trials
            push!(times, @elapsed ŷ = tt_rounding(tts[1] * tts[2] * tts[3]; rmax=tr))
        end
        err = norm(reference - ŷ) / ref_norm
        push!(det[:rk], Float64(tr)); push!(det[:err], err)
        push!(det[:time_med], median(times))
        push!(det[:time_q25], quantile(times, 0.25))
        push!(det[:time_q75], quantile(times, 0.75))
        push!(det[:rk_profile], ŷ.ttv_rks)
        @printf("  rk=%d → err=%.3e, t=%.2fs\n", tr, err, median(times))
        println("         rk profile: ", ŷ.ttv_rks)
    end

    # --- Deterministic baseline at matched tolerance (no rank cap) ---
    println("\n--- Deterministic baseline (matched tol = adaptive ε), inc raw_product cost ---")
    det_tol = Dict{Symbol, Any}(:ε => Float64[], :rk => Float64[], :err => Float64[],
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
        push!(det_tol[:ε], ε)
        push!(det_tol[:rk], Float64(maximum(ŷ.ttv_rks)))
        push!(det_tol[:err], err)
        push!(det_tol[:time_med], median(times))
        push!(det_tol[:time_q25], quantile(times, 0.25))
        push!(det_tol[:time_q75], quantile(times, 0.75))
        push!(det_tol[:rk_profile], ŷ.ttv_rks)
        @printf("  tol=%.0e → max rk=%d, err=%.3e, t=%.2fs\n", ε, maximum(ŷ.ttv_rks), err, median(times))
        println("           rk profile: ", ŷ.ttv_rks)
    end

    # --- ttrand_rounding baseline at each target_rank ---
    println("\n--- ttrand_rounding baseline (block_rks=$block_rks_fixed) ---")
    # JIT warmup
    _ = ttrand_rounding(tts, [1; fill(first(target_ranks), N-1); 1];
                         block_rks=block_rks_fixed, seed=seed)
    rand_b = Dict{Symbol, Any}(:rk => Float64[], :err_med => Float64[],
                               :err_q25 => Float64[], :err_q75 => Float64[],
                               :time_med => Float64[], :time_q25 => Float64[],
                               :time_q75 => Float64[],
                               :rk_profile => Vector{Vector{Int}}())
    for tr in target_ranks
        target_rks = [1; fill(tr, N-1); 1]
        errs = Float64[]; times = Float64[]; rk_profiles = Vector{Int}[]
        for t = 1:n_trials
            local ŷ
            tm = @elapsed ŷ = ttrand_rounding(tts, target_rks;
                                              block_rks=block_rks_fixed,
                                              seed=seed + 100*t)
            push!(errs, norm(reference - ŷ) / ref_norm)
            push!(times, tm)
            push!(rk_profiles, ŷ.ttv_rks)
        end
        push!(rand_b[:rk],      Float64(tr))
        push!(rand_b[:err_med], median(errs))
        push!(rand_b[:err_q25], quantile(errs, 0.25))
        push!(rand_b[:err_q75], quantile(errs, 0.75))
        push!(rand_b[:time_med], median(times))
        push!(rand_b[:time_q25], quantile(times, 0.25))
        push!(rand_b[:time_q75], quantile(times, 0.75))
        med_profile = [round(Int, median([p[k] for p in rk_profiles])) for k = 1:length(rk_profiles[1])]
        push!(rand_b[:rk_profile], med_profile)
        @printf("  rk=%d → median err=%.3e, t=%.2fs\n", tr, median(errs), median(times))
        println("         median rk profile: ", med_profile)
    end

    plot_hadamard_adaptive(adapt, det, det_tol, rand_b; R=R, dir=dir, ref_tol=ref_tol,
                            block_rks_fixed=block_rks_fixed,
                            N=N,
                            ℓ_inc=ℓ_inc,
                            n_trials=n_trials)
    return (adapt=adapt, det=det, det_tol=det_tol, rand=rand_b)
end

function plot_hadamard_adaptive(adapt, det, det_tol, rand_b; R, dir, ref_tol, block_rks_fixed, N, ℓ_inc, n_trials)
    CairoMakie.activate!(type="pdf")
    title_str = "Quantics Hadamard ⊙ benchmark (R=$R, $(n_trials) trials)"

    fig = Figure(size=(1400, 900))
    Label(fig[0, 1:2], title_str; fontsize=13, tellwidth=false)

    # --- Panel 1: error vs rank, all methods on the same axes ---
    ax_re = Axis(fig[1, 1],
                 xlabel="max output rank",
                 ylabel=L"\Vert \hat{x} - x_\mathrm{ref} \Vert / \Vert x_\mathrm{ref} \Vert",
                 yscale=log10,
                 title="error vs achieved rank",
                 titlesize=12)

    scatterlines!(ax_re, det[:rk], max.(det[:err], 1e-18);
                  color=:black, linewidth=2, marker=:circle, markersize=9,
                  label="det @ target rk")
    scatterlines!(ax_re, det_tol[:rk], max.(det_tol[:err], 1e-18);
                  color=:purple, linewidth=2, marker=:star5, markersize=10,
                  label="det @ matched tol")
    band!(ax_re, rand_b[:rk], max.(rand_b[:err_q25], 1e-18), max.(rand_b[:err_q75], 1e-18);
          color=(:gray, 0.30))
    scatterlines!(ax_re, rand_b[:rk], max.(rand_b[:err_med], 1e-18);
                  color=:gray, linewidth=2, marker=:diamond, markersize=9,
                  linestyle=:dash, label="ttrand @ target rk (blk=$block_rks_fixed)")
    band!(ax_re, adapt[:rk_med], max.(adapt[:err_q25], 1e-18), max.(adapt[:err_q75], 1e-18);
          color=(:tomato, 0.25))
    scatterlines!(ax_re, adapt[:rk_med], max.(adapt[:err_med], 1e-18);
                  color=:tomato, linewidth=2, marker=:utriangle, markersize=10,
                  label="adapt (blk=N=$N / N÷4=$(N÷4), τ swept)")
    for (i, ε) in enumerate(adapt[:ε])
        text!(ax_re, adapt[:rk_med][i], max(adapt[:err_med][i], 1e-18);
              text=@sprintf("τ=%.0e", ε), fontsize=8,
              align=(:left, :bottom), offset=(4, 4), color=:tomato)
    end
    axislegend(ax_re; position=:rt, framevisible=true, labelsize=10)

    # --- Panel 2: rank vs tolerance for adaptive (linear y) ---
    ax_rk = Axis(fig[1, 2],
                 xlabel=L"\varepsilon \;(\text{adaptive tolerance})",
                 ylabel="max output rank",
                 xscale=log10, xreversed=true,
                 title="adaptive rank vs tolerance",
                 titlesize=12)
    band!(ax_rk, adapt[:ε], adapt[:rk_q25], adapt[:rk_q75]; color=(:tomato, 0.25))
    scatterlines!(ax_rk, adapt[:ε], adapt[:rk_med];
                  color=:tomato, linewidth=2, marker=:utriangle, markersize=10)
    for (rk, err) in zip(det[:rk], det[:err])
        hlines!(ax_rk, rk; color=(:black, 0.25), linewidth=1, linestyle=:dot)
        text!(ax_rk, last(adapt[:ε]), rk; text="det rk=$(Int(rk))",
              fontsize=8, align=(:left, :bottom), color=(:black, 0.5))
    end

    # --- Panel 3: rank profile per bond, all methods overlaid ---
    ax_pr = Axis(fig[2, 1],
                 xlabel="bond index k",
                 ylabel="TT rank at bond k",
                 yscale=log10,
                 title="per-bond rank profile",
                 titlesize=12)

    # Matched comparison: for each ε, det @ tol=ε (solid blue family) vs adaptive
    # @ τ=ε (dashed red family). Within each family, darker = tighter tolerance.
    @assert length(det_tol[:rk_profile]) == length(adapt[:rk_profile])
    n_ε = length(adapt[:rk_profile])
    det_palette   = cgrad(:Blues, max(4, n_ε + 2))
    adapt_palette = cgrad(:Reds,  max(4, n_ε + 2))
    for i in 1:n_ε
        ε = adapt[:ε][i]
        c_det   = det_palette[i + 2]    # skip the lightest entries
        c_adapt = adapt_palette[i + 2]
        lines!(ax_pr, 1:length(det_tol[:rk_profile][i]), max.(det_tol[:rk_profile][i], 1);
               color=c_det, linewidth=1.8,
               label=@sprintf("det @ tol=%.0e", ε))
        # q25–q75 band for adaptive (trial-to-trial spread of per-bond rank)
        band!(ax_pr, 1:length(adapt[:rk_profile][i]),
              max.(adapt[:rk_profile_q25][i], 1),
              max.(adapt[:rk_profile_q75][i], 1);
              color=(c_adapt, 0.25))
        lines!(ax_pr, 1:length(adapt[:rk_profile][i]), max.(adapt[:rk_profile][i], 1);
               color=c_adapt, linewidth=2.0, linestyle=:dash,
               label=@sprintf("adapt τ=%.0e", ε))
    end
    axislegend(ax_pr; position=:rt, framevisible=true, labelsize=9, nbanks=2)

    # --- Panel 4: wall-time vs achieved error, all methods ---
    ax_t = Axis(fig[2, 2],
                xlabel=L"\Vert \hat{x} - x_\mathrm{ref} \Vert / \Vert x_\mathrm{ref} \Vert",
                ylabel="wall time (s)",
                xscale=log10, yscale=log10, xreversed=true,
                title="wall time vs achieved error",
                titlesize=12)

    band!(ax_t, max.(det[:err], 1e-18),
          max.(det[:time_q25], 1e-4), max.(det[:time_q75], 1e-4);
          color=(:black, 0.20))
    scatterlines!(ax_t, max.(det[:err], 1e-18), max.(det[:time_med], 1e-4);
                  color=:black, linewidth=2, marker=:circle, markersize=9,
                  label="det @ target rk")
    band!(ax_t, max.(det_tol[:err], 1e-18),
          max.(det_tol[:time_q25], 1e-4), max.(det_tol[:time_q75], 1e-4);
          color=(:purple, 0.20))
    scatterlines!(ax_t, max.(det_tol[:err], 1e-18), max.(det_tol[:time_med], 1e-4);
                  color=:purple, linewidth=2, marker=:star5, markersize=10,
                  label="det @ matched tol")
    band!(ax_t, max.(rand_b[:err_med], 1e-18),
          max.(rand_b[:time_q25], 1e-4), max.(rand_b[:time_q75], 1e-4);
          color=(:gray, 0.30))
    scatterlines!(ax_t, max.(rand_b[:err_med], 1e-18), max.(rand_b[:time_med], 1e-4);
                  color=:gray, linewidth=2, marker=:diamond, markersize=9,
                  linestyle=:dash, label="ttrand @ target rk (blk=$block_rks_fixed)")
    band!(ax_t, max.(adapt[:err_med], 1e-18),
          max.(adapt[:time_q25], 1e-4), max.(adapt[:time_q75], 1e-4);
          color=(:tomato, 0.25))
    scatterlines!(ax_t, max.(adapt[:err_med], 1e-18), max.(adapt[:time_med], 1e-4);
                  color=:tomato, linewidth=2, marker=:utriangle, markersize=10,
                  label="adapt (τ swept)")
    for (i, ε) in enumerate(adapt[:ε])
        text!(ax_t, max(adapt[:err_med][i], 1e-18), max(adapt[:time_med][i], 1e-4);
              text=@sprintf("τ=%.0e", ε), fontsize=8,
              align=(:left, :bottom), offset=(4, 4), color=:tomato)
    end
    axislegend(ax_t; position=:lt, framevisible=true, labelsize=10)

    fname = "$(dir)/quantics_hadamard_adaptive_R$(R)_blkfix$(block_rks_fixed)_blkadaptN$(N)i$(N÷4)_linc$(ℓ_inc).pdf"
    save(fname, fig)
    println("\n→ saved $fname")
    return fig
end

run_hadamard_adaptive_benchmark(R=20, ref_tol=1e-10,
                                  εs=(1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8),
                                  block_rks_fixed=8,
                                  ℓ_min=4, ℓ_inc=4, n_trials=10)
