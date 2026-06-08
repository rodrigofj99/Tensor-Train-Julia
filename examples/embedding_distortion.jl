using TensorTrains
using TensorTrains: cached_sketch, ensure_columns!, sketch_matrix
using LinearAlgebra, Statistics, Serialization, Printf, Random
using CairoMakie

"""
Study E — TTStack as a *provable oblivious embedding* for the boundary norm estimate.

The adaptive rounding loop sets its per-bond budget τ from the sketch-based norm estimate
`est = ‖S·y‖_F / ‖y‖` (the boundary contraction of the recursive sketch; Al Daas et al.,
arXiv:2511.03598, Remark 3.1). All sketches are unbiased (E[est²]=1), so the discriminator is the
*distortion / variance* of `est` at a fixed embedding dimension m (= number of sketch columns).

Thesis (CLAUDE.md "oblivious-embedding facts"): the TTStack sketch (`block_rks=N`, within-block QR)
is an oblivious subspace embedding — its distortion depends on the embedding dimension and the
subspace dimension, NOT on the input TT rank. The pure Khatri–Rao product (`block_rks=1`,
`orthogonal=false`) is the data-dependent special case: highest variance, and *worst for rank-1*
inputs (variance improving as the input rank grows). This script quantifies both:

  E1 (mirror of MATLAB NormEstimation.m): mean est ± [min,max] band vs the number of samples, for
     KRP vs TTStack, on each target.
  E2 (the thesis figure): distortion (CoV of est) vs `block_rks`, at fixed embedding dimension,
     overlaid for a rank-1 and a high-rank input — KRP's CoV is high & rank-dependent; raising
     `block_rks` collapses it onto the rank-independent TTStack band.

Run:  <julia> --project=. examples/embedding_distortion.jl            # full
      <julia> --project=. examples/embedding_distortion.jl smoke      # quick check
"""

const MATERN_PATH = joinpath(@__DIR__, "data", "matern_cores.jls")
const OUTDIR = joinpath(@__DIR__, "..", "out", "embedding_distortion")

# ── Targets ─────────────────────────────────────────────────────────────────────────────────────
function load_matern_tt()::TTvector{Float64,8}
    isfile(MATERN_PATH) || error("Missing $MATERN_PATH — see matern_blockrks_sweep.jl docstring.")
    cores = deserialize(MATERN_PATH)
    N = length(cores)
    dims = ntuple(i -> size(cores[i], 2), N)
    rks = vcat(1, [size(c, 3) for c in cores])
    return TTvector{Float64,N}(N, cores, dims, rks, zeros(Int, N))
end

# Synthetic mixed-rank TT (low-rank signal + tiny high-rank perturbation), à la NormEstimation.m
# (defaults match the paper: N=10 modes of size 100, signal/perturbation rank 50, pert 1e-5).
function synthetic_tt(; N=10, d=100, r1=50, r2=50, pert=1e-5, seed=1)
    dims = ntuple(i -> d, N)
    a = rand_tt(Float64, dims, r1; seed=seed)
    b = rand_tt(Float64, dims, r2; seed=seed + 9991)
    return (1/norm(a)) * a + (pert/norm(b)) * b
end

# A rank-`r` TT on the same dims as `ref` (for the rank-contrast in E2).
fixed_rank_tt(ref::TTvector, r::Int; seed=7) = rand_tt(Float64, ref.ttv_dims, r; seed=seed)

# ── Core measurement ─────────────────────────────────────────────────────────────────────────────
# Relative boundary norm estimates est = ‖S·y‖_F/‖y‖ over `nseed` independent sketches, for a single
# pure block-rank group (block_rks_inc = block_rks) grown to ≥ m total sketch columns. `m` is the
# embedding dimension; `block_rks=1, orthogonal=false` is strict KRP, `block_rks=N, orthogonal=true`
# is TTStack.
function norm_est_samples(y::TTvector, block_rks::Int, m::Int, nseed::Int;
                          orthogonal::Bool=true, seed0::Int=1)
    ny = norm(y); N = y.N
    want = fill(m, N + 1); want[N + 1] = 1
    init_rank = max(4, cld(m, 2))
    ests = Vector{Float64}(undef, nseed)
    cols = 0
    for s in 1:nseed
        c = cached_sketch(Float64, y, block_rks, block_rks, init_rank; seed=seed0 + s, orthogonal=orthogonal)
        ensure_columns!(c, y, want; seed=seed0 + s, orthogonal=orthogonal)
        M = sketch_matrix(c, 1, y.ttv_rks[1])      # boundary sketch S·y, shape (1, cols)
        cols = length(M)
        ests[s] = norm(M) / ny
    end
    return ests, cols
end

stats(e) = (mean=mean(e), std=std(e), cov=std(e)/mean(e), lo=minimum(e), hi=maximum(e))

# ── E1: mean ± band vs number of samples (NormEstimation.m mirror) ────────────────────────────────
# methods :: Vector of (label, block_rks, orthogonal); returns Dict label => Dict(:m, :mean, :lo, :hi, :cov)
function run_E1(name::AbstractString, y::TTvector, ms, methods, nseed; dir=OUTDIR)
    @printf("\n[E1] target=%s  N=%d  ranks=%s  ‖y‖=%.4g  (%d seeds)\n",
            name, y.N, string(y.ttv_rks), norm(y), nseed); flush(stdout)
    res = Dict{String,Dict{Symbol,Vector{Float64}}}()
    for (label, brk, orth) in methods
        d = Dict(:m=>Float64[], :cols=>Float64[], :mean=>Float64[], :lo=>Float64[], :hi=>Float64[], :cov=>Float64[])
        for m in ms
            e, cols = norm_est_samples(y, brk, m, nseed; orthogonal=orth)
            st = stats(e)
            push!(d[:m], m); push!(d[:cols], cols); push!(d[:mean], st.mean)
            push!(d[:lo], st.lo); push!(d[:hi], st.hi); push!(d[:cov], st.cov)
            @printf("   %-12s m=%3d cols=%3d  mean=%.4f  cov=%.4f  band=[%.3f,%.3f]\n",
                    label, m, cols, st.mean, st.cov, st.lo, st.hi); flush(stdout)
        end
        res[label] = d
    end
    mkpath(dir); serialize(joinpath(dir, "E1_$(name).jls"), res)
    return res
end

# ── E2: distortion (CoV) vs block_rks at fixed embedding dim, per input-rank ──────────────────────
# targets :: Vector of (rank_label, y); returns Dict rank_label => Dict(:brk, :cov, :mean)
function run_E2(name::AbstractString, targets, brks, m, nseed; dir=OUTDIR, orthogonal=true)
    @printf("\n[E2] %s  embedding m=%d  block_rks∈%s  (%d seeds)\n",
            name, m, string(collect(brks)), nseed); flush(stdout)
    res = Dict{String,Dict{Symbol,Vector{Float64}}}()
    for (rlabel, y) in targets
        d = Dict(:brk=>Float64[], :cov=>Float64[], :mean=>Float64[])
        for brk in brks
            e, _ = norm_est_samples(y, brk, m, nseed; orthogonal=orthogonal)
            st = stats(e)
            push!(d[:brk], brk); push!(d[:cov], st.cov); push!(d[:mean], st.mean)
            @printf("   %-16s block_rks=%2d  cov=%.4f  mean=%.4f\n", rlabel, brk, st.cov, st.mean); flush(stdout)
        end
        res[rlabel] = d
    end
    mkpath(dir); serialize(joinpath(dir, "E2_$(name).jls"), res)
    return res
end

# ── Plots ─────────────────────────────────────────────────────────────────────────────────────────
const _E_COLORS = [:tomato, :seagreen, :royalblue, :purple, :darkorange, :firebrick]

function plot_E1(name, res, order; dir=OUTDIR)
    CairoMakie.activate!(type="pdf"); mkpath(dir)
    fig = Figure(size=(900, 420))
    Label(fig[0, 1:2], "E1 — norm estimate ‖S·y‖/‖y‖ vs samples ($name)"; fontsize=13, tellwidth=false)
    ax1 = Axis(fig[1, 1], xlabel="sketch columns (embedding dim)", ylabel="estimate / ‖y‖",
               title="mean ± [min,max] band", titlesize=11)
    hlines!(ax1, [1.0]; color=:black, linestyle=:dash, linewidth=1)
    for (i, lbl) in enumerate(order)
        d = res[lbl]; c = _E_COLORS[mod1(i, length(_E_COLORS))]
        band!(ax1, d[:cols], d[:lo], d[:hi]; color=(c, 0.18))
        scatterlines!(ax1, d[:cols], d[:mean]; color=c, linewidth=2, markersize=7, label=lbl)
    end
    axislegend(ax1; position=:rt, labelsize=9)
    ax2 = Axis(fig[1, 2], xlabel="sketch columns (embedding dim)", ylabel="CoV of estimate",
               yscale=log10, title="distortion (coefficient of variation)", titlesize=11)
    for (i, lbl) in enumerate(order)
        d = res[lbl]; c = _E_COLORS[mod1(i, length(_E_COLORS))]
        scatterlines!(ax2, d[:cols], max.(d[:cov], 1e-6); color=c, linewidth=2, markersize=7, label=lbl)
    end
    axislegend(ax2; position=:rt, labelsize=9)
    f = joinpath(dir, "E1_$(name).pdf"); save(f, fig); println("→ saved $f"); return fig
end

function plot_E2(name, res, order, m; dir=OUTDIR)
    CairoMakie.activate!(type="pdf"); mkpath(dir)
    fig = Figure(size=(640, 460))
    Label(fig[0, 1], "E2 — distortion vs block rank at fixed embedding (m≈$m, $name)";
          fontsize=13, tellwidth=false)
    ax = Axis(fig[1, 1], xlabel="block_rks (1 = KRP … N = TTStack)", ylabel="CoV of ‖S·y‖/‖y‖",
              yscale=log10, xscale=log2, title="oblivious embedding: CoV → rank-independent band", titlesize=11)
    for (i, lbl) in enumerate(order)
        d = res[lbl]; c = _E_COLORS[mod1(i, length(_E_COLORS))]
        scatterlines!(ax, d[:brk], max.(d[:cov], 1e-6); color=c, linewidth=2, markersize=8, label=lbl)
    end
    axislegend(ax; position=:rt, labelsize=10)
    f = joinpath(dir, "E2_$(name).pdf"); save(f, fig); println("→ saved $f"); return fig
end

# ── Drivers ──────────────────────────────────────────────────────────────────────────────────────
function run_synthetic(; nseed=1000, smoke=false)
    Ns = smoke ? (10,) : (5, 10, 15)
    ms = smoke ? (10, 40) : (10:20:200)
    methods = [("strict-KRP", 1, false), ("orth-KRP", 1, true), ("TTStack-N", :N, true)]
    for Nm in Ns
        y = synthetic_tt(N=Nm, d=100, r1=50, r2=50, pert=1e-5, seed=1)
        meths = [(l, b === :N ? Nm : b, o) for (l, b, o) in methods]
        res = run_E1("synthetic_N$(Nm)", y, ms, meths, nseed)
        plot_E1("synthetic_N$(Nm)", res, [m[1] for m in meths])
    end
    # E2 rank contrast on N=10 dims: rank-1 vs rank-50 (signal) vs synthetic(50+pert).
    if !smoke
        base = synthetic_tt(N=10, d=100, seed=1)
        targets = [("rank-1", fixed_rank_tt(base, 1)),
                   ("rank-10", fixed_rank_tt(base, 10)),
                   ("rank-50", fixed_rank_tt(base, 50))]
        m = 96
        res2 = run_E2("synthetic", targets, (1, 2, 4, 8, 16, 32), m, max(200, nseed ÷ 5))
        plot_E2("synthetic", res2, [t[1] for t in targets], m)
    end
end

function run_matern_targets(; nseed=150)
    y = load_matern_tt()                                  # rank-621
    methods = [("strict-KRP", 1, false), ("orth-KRP", 1, true), ("TTStack-N", y.N, true)]
    res = run_E1("matern", y, 10:20:170, methods, nseed)
    plot_E1("matern", res, [m[1] for m in methods])
    # E2: rank-1 vs rank-621 on the Matérn dims — the headline rank-independence contrast.
    targets = [("rank-1", fixed_rank_tt(y, 1)), ("Matérn (rank-621)", y)]
    m = 96
    res2 = run_E2("matern_rankcontrast", targets, (1, 2, 4, 8, 16, 32), m, nseed)
    plot_E2("matern_rankcontrast", res2, [t[1] for t in targets], m)
end

function run_cookie_target(; nseed=150)
    # Build a representative cookie TT (the normalized RHS, a rank-1 TT) and a mid-rank cookie iterate.
    include(joinpath(@__DIR__, "cookie_gmres.jl"))
    A0, A_c, a0 = Base.invokelatest(load_cookie_data)
    op = Base.invokelatest(CookieOp, A0, A_c, 8; minD=1.0, maxD=5.0)
    b = Base.invokelatest(cookie_rhs, op, a0); b = (1/norm(b)) * b
    targets = [("cookie-RHS (rank-1)", b)]
    res2 = run_E2("cookie", targets, (1, 2, 4, 8, 16, 32), 96, nseed)
    plot_E2("cookie", res2, [t[1] for t in targets], 96)
end

function main(; smoke=false)
    BLAS.set_num_threads(1)   # determinism for variance estimates
    if smoke
        run_synthetic(nseed=20, smoke=true)
        return
    end
    run_synthetic(nseed=1000)
    run_matern_targets(nseed=150)
    run_cookie_target(nseed=150)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(smoke = ("smoke" in ARGS))
end
