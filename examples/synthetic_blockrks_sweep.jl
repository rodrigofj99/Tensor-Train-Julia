using TensorTrains
using LinearAlgebra, Statistics, Serialization, Printf, Random
using CairoMakie

include(joinpath(@__DIR__, "sweep_common.jl"))

"""
Study S — synthetic-tensor fixed-rank randomized rounding (paper §4.1 / SyntheticTest.m mirror), with
the KRP→TTStack block-rank contrast.

Target (paper §4.1): a synthetic TT `X = a/‖a‖ + 1e-5·b/‖b‖`, with `a`, `b` random rank-50 TTs over
N=10 modes of size 100. Sweep the maximum target rank `rmax ∈ 35:5:80` and compare the fixed-rank
randomized rounding `ttrand_rounding(X, rmax; block_rks, orthogonal)` against the deterministic
`tt_rounding(X; rmax)` baseline. Metrics: relative error `‖X−X̂‖/‖X‖` and speedup `t_det/t_rand`.

Methods (named vocabulary, shared with the other studies):
  strict-KRP  = block_rks=1, orthogonal=false   (raw Khatri–Rao, the paper's Rand-Orth-KRP)
  orth-KRP    = block_rks=1, orthogonal=true
  TTStack-N   = block_rks=N, orthogonal=true     (oblivious embedding)
  TTStack-2N  = block_rks=2N, orthogonal=true

Run:  <julia> --project=. examples/synthetic_blockrks_sweep.jl          # full
      <julia> --project=. examples/synthetic_blockrks_sweep.jl smoke    # quick check
"""

const OUTDIR = joinpath(@__DIR__, "..", "out", "synthetic_blockrks_sweep")

function synthetic_tt(; N=10, d=100, r1=50, r2=50, pert=1e-5, seed=1)
    dims = ntuple(i -> d, N)
    a = rand_tt(Float64, dims, r1; seed=seed)
    b = rand_tt(Float64, dims, r2; seed=seed + 9991)
    return (1/norm(a)) * a + (pert/norm(b)) * b
end

function run_synthetic_sweep(; rmaxs = 35:5:80, n_trials = 10, seed = 1234, smoke = false,
                              dir = OUTDIR)
    smoke && (rmaxs = (40, 60); n_trials = 2)
    mkpath(dir)
    X = synthetic_tt()
    N = X.N; nX = norm(X)
    @printf("Study S: synthetic TT  N=%d  dims=%d  ranks=%s  ‖X‖=%.4g\n",
            N, X.ttv_dims[1], string(X.ttv_rks), nX)

    variants = [(label="strict-KRP", blk=1,   orth=false),
                (label="orth-KRP",   blk=1,   orth=true),
                (label="TTStack-N",  blk=N,   orth=true),
                (label="TTStack-2N", blk=2N,  orth=true)]

    # Deterministic baseline per rmax (error + time).
    det = Dict(:rmax=>Float64[], :err=>Float64[], :time_med=>Float64[])
    println("\n--- deterministic baseline ---"); flush(stdout)
    for r in rmaxs
        ts = Float64[]; local x̂
        for _ in 1:max(2, n_trials ÷ 2); push!(ts, @elapsed x̂ = tt_rounding(X; rmax=r)); end
        push!(det[:rmax], r); push!(det[:err], norm(X - x̂)/nX); push!(det[:time_med], median(ts))
        @printf("  rmax=%2d  err=%.3e  t=%.3fs\n", r, norm(X - x̂)/nX, median(ts)); flush(stdout)
    end
    dump_incremental(dir, "S_det.jls", det)

    adapt = Dict{String,Dict{Symbol,Vector{Float64}}}()
    for v in variants
        println("\n--- $(v.label) (block_rks=$(v.blk), orth=$(v.orth)) ---"); flush(stdout)
        # warmup (runtime block_rks/orth → one warmup compiles the path)
        _ = ttrand_rounding(X, first(rmaxs); block_rks=v.blk, orthogonal=v.orth, seed=seed)
        d = Dict(:rmax=>Float64[], :err_med=>Float64[], :err_q25=>Float64[], :err_q75=>Float64[],
                 :time_med=>Float64[], :time_q25=>Float64[], :time_q75=>Float64[], :speedup=>Float64[])
        for (ri, r) in enumerate(rmaxs)
            errs = Float64[]; times = Float64[]
            for t in 1:n_trials
                local x̂
                tm = @elapsed x̂ = ttrand_rounding(X, r; block_rks=v.blk, orthogonal=v.orth, seed=seed + 1000t)
                push!(errs, norm(X - x̂)/nX); push!(times, tm)
            end
            es = trial_stats(errs); tss = trial_stats(times)
            push!(d[:rmax], r); push!(d[:err_med], es.med); push!(d[:err_q25], es.q25); push!(d[:err_q75], es.q75)
            push!(d[:time_med], tss.med); push!(d[:time_q25], tss.q25); push!(d[:time_q75], tss.q75)
            push!(d[:speedup], det[:time_med][ri] / tss.med)
            @printf("  rmax=%2d  err=%.3e  t=%.3fs  speedup=%.2f\n", r, es.med, tss.med, det[:time_med][ri]/tss.med); flush(stdout)
        end
        adapt[v.label] = d
        dump_incremental(dir, "S_adapt_$(v.label).jls", d)
    end
    dump_incremental(dir, "S_all.jls", (adapt=adapt, det=det))
    plot_synthetic_sweep(adapt, det, [v.label for v in variants]; dir=dir, n_trials=n_trials)
    return adapt, det
end

function plot_synthetic_sweep(adapt, det, labels; dir, n_trials)
    CairoMakie.activate!(type="pdf"); mkpath(dir)
    fig = Figure(size=(1100, 460))
    Label(fig[0, 1:2], "Study S — synthetic rank-50+1e-5 TT, fixed-rank rounding ($(n_trials) trials)";
          fontsize=13, tellwidth=false)
    ax_e = Axis(fig[1, 1], xlabel="max target rank", ylabel="rel. error ‖X−X̂‖/‖X‖",
                yscale=log10, title="accuracy vs target rank", titlesize=12)
    for lbl in labels
        d = adapt[lbl]; c, m = variant_style(lbl)
        band!(ax_e, d[:rmax], max.(d[:err_q25],1e-18), max.(d[:err_q75],1e-18); color=(c,0.22))
        scatterlines!(ax_e, d[:rmax], max.(d[:err_med],1e-18); color=c, linewidth=2, marker=m, markersize=9, label=lbl)
    end
    c, m = variant_style("det")
    scatterlines!(ax_e, det[:rmax], max.(det[:err],1e-18); color=c, linewidth=2, marker=m, markersize=10, label="det @ rmax")
    axislegend(ax_e; position=:lb, framevisible=true, labelsize=10)

    ax_s = Axis(fig[1, 2], xlabel="max target rank", ylabel="speedup (t_det / t_rand)",
                title="speedup vs deterministic", titlesize=12)
    hlines!(ax_s, [1.0]; color=:black, linestyle=:dash, linewidth=1)
    for lbl in labels
        d = adapt[lbl]; c, m = variant_style(lbl)
        scatterlines!(ax_s, d[:rmax], d[:speedup]; color=c, linewidth=2, marker=m, markersize=9, label=lbl)
    end
    axislegend(ax_s; position=:rt, framevisible=true, labelsize=10)
    f = joinpath(dir, "synthetic_blockrks_sweep.pdf"); save(f, fig); println("\n→ saved $f"); return fig
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_synthetic_sweep(smoke = ("smoke" in ARGS))
end
