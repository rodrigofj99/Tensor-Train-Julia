using TensorTrains
using LinearAlgebra
using Random
using Statistics
using Printf
using CairoMakie
using Serialization

include(joinpath(@__DIR__, "sweep_common.jl"))

"""
Block-rank-extension sweep on the Matérn kernel TT tensor from §4.2 of
Al Daas et al., arXiv:2511.03598 (paper-companion repo
https://github.com/bhisham123/TT-Rounding-using-KRP-structure, MaternTest.m).

Input is the pre-computed 8-core TT representation of an 8-way Matérn
kernel coefficient tensor at Chebyshev nodes, n=100 per mode, TT-cross
compressed to ~1e-12 tolerance. Ranks `(1,12,67,257,621,259,65,12,1)`.

The benchmark is the single-TT analogue of `quantics_hadamard_blockrks_sweep.jl`
— five adaptive variants (one KRP, four `N init / ext=*`) vs the
deterministic `tt_rounding(...; tol=ε)`. Same plot layout (4 panels).

Data file `examples/data/matern_cores.jls` is git-ignored; obtain it by
downloading `matern_data.mat` from the paper's repo (Git LFS, ~285 MB)
and converting via the steps documented in the accompanying README.
"""

const DATA_PATH = joinpath(@__DIR__, "data", "matern_cores.jls")

function load_matern_tt()::TTvector{Float64,8}
    isfile(DATA_PATH) || error("Missing $DATA_PATH — see top-of-file docstring for how to produce it.")
    cores = deserialize(DATA_PATH)
    N = length(cores)
    # Cores serialized in (L, I, R) layout — matches TTvector's post-refactor convention.
    dims = ntuple(i -> size(cores[i], 2), N)
    rks = vcat(1, [size(c, 3) for c in cores])
    ot = zeros(Int, N)
    return TTvector{Float64,N}(N, cores, dims, rks, ot)
end

function run_matern_sweep(; ref_tol::Float64 = 1e-10,
                            εs = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7),
                            n_trials::Int = 50,
                            seed::Int = 1234,
                            dir::String = "out/matern_blockrks_sweep")
    mkpath(dir)
    println("Loading Matérn kernel TT from $DATA_PATH …")
    y = load_matern_tt()
    N = y.N
    # init_f=0.1 sets the per-bond INITIAL sketch width (10% of the bond's rank
    # cap). ℓ_inc is only a small absolute FLOOR on the per-iteration increment:
    # the increment grows locally inside the algorithm as 0.2·current_cols, so the
    # effective step scales with the *local* bond rank, not the global ℓ_max.
    # (Tying ℓ_inc to ℓ_max over-inflates mid-bond ranks where ℓ_max ≫ the
    # typical bond rank — e.g. Kronecker/Hadamard products.)
    ℓ_inc = 4
    n_samples = max(ℓ_inc, 20)
    init_f = 0.1
    println("  N=$N, mode size=$(y.ttv_dims[1]), input ranks=$(y.ttv_rks)")
    println("  params: init_f=$init_f  ℓ_inc(floor)=$ℓ_inc  n_samples=$n_samples  (increment grows locally as 0.2·rank)")
    reference = tt_rounding(y; tol=ref_tol)
    println("  Reference (det @ tol=$ref_tol) max rk = $(maximum(reference.ttv_rks))")
    ref_norm = norm(reference)
    println("  ‖ref‖ = $(round(ref_norm, sigdigits=6))")

    # `orth`: whether to QR-orthogonalize the per-block sketch vectors.
    #   `false` matches MATLAB's KRP (raw Gaussian, no normalisation) — Khatri-Rao
    #     in the strict sense of the original paper.
    #   `true` is our TTStack-style orthogonalised sketch (within-block QR).
    # Named-method grid (shared vocabulary). Sweeps block_rks (init group: 1=KRP … 2N) AND
    # block_rks_inc (ext group), plus the orthogonal=false strict-KRP contrast and the Adap-R column
    # (TTStack-N followed by a final deterministic tt_rounding(;tol=ε) pass — paper Remark 3.2).
    variants = (
      (label="strict-KRP", blk=1,  inc=1, orth=false, adap_r=false),
      (label="orth-KRP",   blk=1,  inc=1, orth=true,  adap_r=false),
      (label="TTStack-N",  blk=N,  inc=4, orth=true,  adap_r=false),
      (label="TTStack-2N", blk=2N, inc=8, orth=true,  adap_r=false),
      (label="Adap-R",     blk=N,  inc=4, orth=true,  adap_r=true),
    )

    println("\n--- Warmup ---")
    for v in variants
        _ = ttrand_rounding_adaptive(y, first(εs); ℓ_min=1, init_f=init_f, ℓ_inc=ℓ_inc,
                                      n_samples=n_samples, orthogonal=v.orth,
                                      block_rks=v.blk, block_rks_inc=v.inc, seed=seed)
    end
    _ = tt_rounding(y; tol=first(εs))

    adapt_results = Dict{String, Dict{Symbol, Any}}()
    for v in variants
        println("\n--- Adaptive sweep $(v.label) (block_rks=$(v.blk), block_rks_inc=$(v.inc), orth=$(v.orth), ℓ_inc=$ℓ_inc) ---")
        flush(stdout)
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
                tm = @elapsed begin
                    ŷ = ttrand_rounding_adaptive(y, ε;
                                                 ℓ_min=1, init_f=init_f, ℓ_inc=ℓ_inc,
                                                 n_samples=n_samples,
                                                 orthogonal=v.orth,
                                                 block_rks=v.blk,
                                                 block_rks_inc=v.inc,
                                                 seed=seed + 1000*t)
                    # Adap-R (Remark 3.2): a final deterministic rounding pass to shed excess rank;
                    # its cost is included in the timing so the speedup/compression are fair.
                    v.adap_r && (ŷ = tt_rounding(ŷ; tol=ε))
                end
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
            flush(stdout)
            # Incremental snapshot so a crash mid-sweep doesn't lose all prior work
            serialize(joinpath(dir, "partial_adapt_$(replace(v.label, '/'=>'_'))_ε$ε.jls"),
                      (label=v.label, ε=ε, errs=errs, rks=rks, times=times, rk_profiles=rk_profiles))
        end
        adapt_results[v.label] = d
        # Full per-variant dump after its inner loop finishes
        serialize(joinpath(dir, "adapt_$(replace(v.label, '/'=>'_')).jls"), d)
        flush(stdout)
    end
    # All adaptive variants done — dump aggregate
    serialize(joinpath(dir, "adapt_all.jls"), adapt_results)

    println("\n--- Deterministic baseline (matched tol) ---")
    det = Dict{Symbol, Any}(:ε => Float64[], :rk => Float64[], :err => Float64[],
                            :time_med => Float64[], :time_q25 => Float64[],
                            :time_q75 => Float64[],
                            :rk_profile => Vector{Vector{Int}}())
    for ε in εs
        local ŷ
        times = Float64[]
        for _ in 1:n_trials
            push!(times, @elapsed ŷ = tt_rounding(y; tol=ε))
        end
        err = norm(reference - ŷ) / ref_norm
        push!(det[:ε], ε); push!(det[:rk], Float64(maximum(ŷ.ttv_rks)))
        push!(det[:err], err)
        push!(det[:time_med], median(times))
        push!(det[:time_q25], quantile(times, 0.25))
        push!(det[:time_q75], quantile(times, 0.75))
        push!(det[:rk_profile], ŷ.ttv_rks)
        @printf("  tol=%.0e → max rk=%d, err=%.3e, t=%.3fs\n",
                ε, maximum(ŷ.ttv_rks), err, median(times))
    end
    serialize(joinpath(dir, "det.jls"), det)

    variant_labels = [v.label for v in variants]
    title = "Matérn kernel TT — block-rank study (N=$N, $(n_trials) trials, ℓ_inc=$ℓ_inc)"
    four_panel_sweep(adapt_results, det; variant_labels=variant_labels, N=N, n_trials=n_trials,
                     title=title, dir=dir, fname="matern_blockrks_ext_sweep.pdf")
    speedup_compression_panel(adapt_results, det; variant_labels=variant_labels,
                              title="Matérn — accuracy / speedup / compression vs tolerance (paper Fig 5)",
                              dir=dir, fname="matern_speedup_compression.pdf")
    return (adapt=adapt_results, det=det)
end


if abspath(PROGRAM_FILE) == @__FILE__
    run_matern_sweep(n_trials=5)
end
