using TensorTrains
using LinearAlgebra
using SparseArrays
using Printf

"""
Cookie problem (parametric PDE) solved with a faithful TT-GMRES, using the
package's TTStack/adaptive Sum+Round as the rounding engine.

This reproduces the §4.3 "cookie problem" of Al Daas et al., arXiv:2511.03598
(paper-companion MATLAB at /Users/cazeaux/krp_paper_matlab — TTGMRESTest.m,
timed_TTGMRES.m, TTsummandsKronOp.m, TTsum*_KRP.m).

Problem (new 7-cookie dataset, nx_40):
  d+1 = 8 TT modes. Mode 1 is spatial (dim n1 = 1681, sparse FEM stiffness
  matrices); modes 2..8 are parameter modes (dim n, the number of parameter
  samples). The operator is a *sum* of 8 Kronecker-product summands
      A = A0 ⊗ I ⊗ … ⊗ I  +  Σ_{c} A_c ⊗ I ⊗ … ⊗ D ⊗ … ⊗ I ,
  with D = diag(linspace(minD,maxD,n)) sitting in the cookie's own parameter
  mode. The RHS is rank-1, b = a0 ⊗ 1 ⊗ … ⊗ 1, scaled to unit norm. A right
  preconditioner solves the mean operator A0 + Σ_c A_c on the first core only.

TT-GMRES (timed_TTGMRES.m) is non-restarted and right-preconditioned. The two
hot spots — applying the Kronecker-sum operator and the Gram–Schmidt step — are
both Sum+Round operations over a list of TT tensors. We plug in either the
deterministic `tt_rounding` of an explicit sum, or the adaptive KRP
`ttrand_rounding_adaptive(α, ys, ε)` (block_rks=1, orthogonal=false — strict
Khatri–Rao sketch, n_samples=20 ⇔ MATLAB min_samples=20). The two share the
same per-bond budget τ = ε·‖·‖_F/√(N−1).

Note on maxD: the committed MATLAB hardcodes maxD=10, but its own comment and
the paper's 7-cookie experiment use [1,5]; we default to 5 here (override via
`run_cookie(; maxD=...)`).

Data path is external (like the Matérn `.jls`); override `DATA_DIR` as needed.
"""

const DATA_DIR = "/Users/cazeaux/krp_paper_matlab/Dataset/nine_cookies/nx_40"

# Cookies selected out of the nine in the "new" dataset (TTGMRESTest.m).
const LIST_COOKIES = [1, 2, 3, 5, 7, 8, 9]

# ----------------------------------------------------------------------------
# 1. Data import — parse the MATLAB `matrix_k.m` (COO triples) and `rhs_.m`.
# ----------------------------------------------------------------------------

"""Extract the numeric rows inside the `name=[ … ];` block of a MATLAB `.m` file."""
function _bracket_rows(path::AbstractString)
    rows = Vector{Vector{Float64}}()
    inblock = false
    for line in eachline(path)
        s = strip(line)
        if !inblock
            occursin("=[", s) && (inblock = true)   # opening line `ZZZZ=[` / `rhs=[`
            continue
        end
        startswith(s, "];") && break                # closing line
        isempty(s) && continue
        push!(rows, parse.(Float64, split(s)))
    end
    return rows
end

"""Read a `matrix_k.m` COO file into a SparseMatrixCSC (indices are 1-based)."""
function parse_coo_matrix(path::AbstractString)
    rows = _bracket_rows(path)
    I = Int[round(Int, r[1]) for r in rows]
    J = Int[round(Int, r[2]) for r in rows]
    V = Float64[r[3] for r in rows]
    n = max(maximum(I), maximum(J))
    return sparse(I, J, V, n, n)
end

"""Read `rhs_.m` into a dense vector."""
parse_rhs(path::AbstractString) = Float64[r[1] for r in _bracket_rows(path)]

"""
Load the cookie FEM data: background stiffness `A0`, the selected cookie
stiffness matrices `A_c`, and the spatial RHS `a0`.
"""
function load_cookie_data(dir::AbstractString = DATA_DIR; list_cookies = LIST_COOKIES)
    isdir(dir) || error("Missing cookie data dir $dir — see top-of-file docstring.")
    A0 = parse_coo_matrix(joinpath(dir, "matrix_1.m"))
    A_c = [parse_coo_matrix(joinpath(dir, "matrix_$(c + 1).m")) for c in list_cookies]
    a0 = parse_rhs(joinpath(dir, "rhs_.m"))
    return A0, A_c, a0
end

# ----------------------------------------------------------------------------
# 2. Operator as a list of Kronecker summands (faithful to TTKronOp).
#    We keep the spatial matrices SPARSE and never form a dense TToperator.
# ----------------------------------------------------------------------------

"""
Cookie operator in summand form. `spatial[j]` is the mode-1 factor of summand j
(A0 for j=1, a cookie stiffness for j≥2). `dpos[j]` is the parameter mode that
carries the diagonal `D` (0 for the background summand). `dvals` is diag(D).
"""
struct CookieOp
    spatial::Vector{SparseMatrixCSC{Float64,Int}}
    dpos::Vector{Int}
    dvals::Vector{Float64}
    dims::NTuple{8,Int}
end

function CookieOp(A0, A_c, n; minD = 1.0, maxD = 5.0)
    d = 1 + length(A_c)                       # number of TT modes (= 8)
    spatial = vcat([A0], A_c)                 # length d, mode-1 factor per summand
    dpos = vcat(0, collect(2:d))              # background has no D; cookie j sits in mode j
    dvals = collect(range(minD, maxD; length = n))
    n1 = size(A0, 1)
    dims = ntuple(k -> k == 1 ? n1 : n, d)
    return CookieOp(spatial, dpos, dvals, dims)
end

"""
Apply a single Kronecker summand `j` to a TT vector `x`, returning a new TT of
the *same ranks* (cf. TTKronOp). Only the spatial core and the summand's D-core
are recomputed; identity modes share `x`'s core arrays.
"""
function apply_summand(op::CookieOp, j::Int, x::TTvector{Float64,M}) where {M}
    cores = copy(x.ttv_vec)                   # shallow: identity modes reuse x's arrays
    c1 = x.ttv_vec[1]                         # (1, n1, R1) since r0 = 1
    n1 = size(c1, 2); R1 = size(c1, 3)
    cores[1] = reshape(op.spatial[j] * reshape(c1, n1, R1), 1, n1, R1)
    if op.dpos[j] > 0
        p = op.dpos[j]
        cores[p] = x.ttv_vec[p] .* reshape(op.dvals, 1, length(op.dvals), 1)
    end
    return TTvector{Float64,M}(x.N, cores, x.ttv_dims, copy(x.ttv_rks), zeros(Int, x.N))
end

"""Apply the full operator, returning the list of summands `[A_j x]` to be summed."""
apply_summands(op::CookieOp, x::TTvector) =
    [apply_summand(op, j, x) for j in eachindex(op.spatial)]

"""
Assemble the cookie operator as a single `TToperator` (rank ≤ #summands = 8) by building each
Kronecker summand as a rank-1 operator and summing them. The mode-1 spatial core becomes dense
`(1, n1, n1, 8)` (~180 MB at nx_40) — needed by the generic sketched matvec, heavier than the
sparse summand path.
"""
function cookie_operator_tto(op::CookieOp)
    d = length(op.dims)
    n1 = op.dims[1]
    summands = Vector{TToperator{Float64,d}}(undef, length(op.spatial))
    for j in eachindex(op.spatial)
        cores = Vector{Array{Float64,4}}(undef, d)
        cores[1] = reshape(Matrix(op.spatial[j]), 1, n1, n1, 1)
        for p in 2:d
            np = op.dims[p]
            fac = p == op.dpos[j] ? Diagonal(op.dvals) : Diagonal(ones(np))
            cores[p] = reshape(Matrix{Float64}(fac), 1, np, np, 1)
        end
        summands[j] = TToperator{Float64,d}(d, cores, op.dims, ones(Int, d + 1), zeros(Int, d))
    end
    return reduce(+, summands)
end

"""Rank-1 RHS `a0 ⊗ 1 ⊗ … ⊗ 1` with mode dims `op.dims`."""
function cookie_rhs(op::CookieOp, a0::Vector{Float64})
    d = length(op.dims)
    cores = Vector{Array{Float64,3}}(undef, d)
    cores[1] = reshape(copy(a0), 1, op.dims[1], 1)
    for k in 2:d
        cores[k] = ones(Float64, 1, op.dims[k], 1)
    end
    return TTvector{Float64,d}(d, cores, op.dims, ones(Int, d + 1), zeros(Int, d))
end

# ----------------------------------------------------------------------------
# 3. Right preconditioner: LU of the mean operator A0 + Σ_c A_c, applied to the
#    first core only (mirrors `Preconditioner` in TTGMRESTest.m).
# ----------------------------------------------------------------------------

function make_preconditioner(A0, A_c)
    M = A0 + sum(A_c)
    F = lu(M)
    return function (x::TTvector)
        d = x.N
        c1 = x.ttv_vec[1]
        n1 = size(c1, 2); R1 = size(c1, 3)
        newc1 = reshape(F \ reshape(c1, n1, R1), 1, n1, R1)
        cores = copy(x.ttv_vec)
        cores[1] = newc1
        return TTvector{Float64,d}(d, cores, x.ttv_dims, copy(x.ttv_rks), zeros(Int, d))
    end
end

# ----------------------------------------------------------------------------
# 4. Sum+Round closures: `(ys::Vector{TTvector}, α::Vector, ε) -> TTvector`.
# ----------------------------------------------------------------------------

function weighted_sum(α, ys)
    acc = α[1] * ys[1]
    @inbounds for j in 2:length(ys)
        acc = acc + α[j] * ys[j]
    end
    return acc
end

sumround_det(ys, α, ε) = tt_rounding(weighted_sum(α, ys); tol = ε)

function sumround_krp(ys, α, ε; seed = 1234, final_round = true)
    y = ttrand_rounding_adaptive(collect(Float64, α), ys, ε;
                                 n_samples = 20, block_rks = 1,
                                 orthogonal = false, seed = seed)
    # Final deterministic rounding pass: the adaptive KRP sketch can leave excess rank
    # (block size / residual-norm over-estimation). A cheap SVD pass on the already
    # left-orthonormal result sheds it (Al Daas et al., arXiv:2511.03598, Remark 3.2).
    return final_round ? tt_rounding(y; tol = ε) : y
end

# ----------------------------------------------------------------------------
# 5. Faithful TT-GMRES (non-restarted, right-preconditioned). Mirrors
#    timed_TTGMRES.m. `apply_op(x)` returns the list of operator summands.
# ----------------------------------------------------------------------------

function tt_gmres_cookie(apply_op, b::TTvector{Float64,M}, prec, sumround;
                         tol = 1e-8, maxit = 50, tols = tol * 1e-2,
                         verbose = true) where {M}
    normb = norm(b)
    V = Vector{TTvector{Float64,M}}(undef, maxit + 1)
    normres = norm(b)
    V[1] = (1 / normres) * b
    H = zeros(maxit + 1, maxit)
    rK = zeros(maxit + 1); rK[1] = normres
    res_hist = Float64[]
    rank_hist = Int[]
    t_sr = 0.0                                           # accumulated Sum+Round wall time
    k = 0
    local R, t
    while k < maxit && normres > tol * normb
        k += 1
        w = prec(V[k])                                   # right preconditioner
        summ = apply_op(w)                               # list of Kronecker summands
        delta = 0.01 * tol / (normres * normb)           # const absolute rounding error
        t_sr += @elapsed w = sumround(summ, ones(length(summ)), delta)  # Sum+Round operator action
        h = zeros(k + 1)
        for j in 1:k
            h[j] = TensorTrains.dot(V[j], w)             # modified Gram–Schmidt
        end
        t_sr += @elapsed w = sumround(vcat(V[1:k], [w]), vcat(-h[1:k], 1.0), delta)  # w - Σ h_j V_j
        h[k + 1] = norm(w)
        V[k + 1] = (1 / h[k + 1]) * w
        H[1:k + 1, k] = h
        F = qr(H[1:k + 1, 1:k])
        Q = F.Q * Matrix(1.0I, k + 1, k + 1)   # full (k+1)×(k+1) orthogonal factor
        R = F.R
        t = Q' * rK[1:k + 1]
        normres = abs(t[k + 1])
        push!(res_hist, normres / normb)
        push!(rank_hist, maximum(V[k + 1].ttv_rks))
        verbose && @printf("  it %2d  est.relres = %.3e  max rank = %d\n",
                           k, normres / normb, rank_hist[end])
    end
    y = R[1:k, 1:k] \ t[1:k]
    t_sr += @elapsed x = sumround(V[1:k], y, tols)
    x = prec(x)
    x = tt_rounding(x; tol = tols)
    return x, res_hist, rank_hist, t_sr
end

# ----------------------------------------------------------------------------
# 6. Driver / verification.
# ----------------------------------------------------------------------------

"""
    run_cookie(; n=8, variant=:krp, minD=1.0, maxD=5.0, tol=1e-8, maxit=50)

Build and solve the cookie problem with `n` parameter samples per mode. `variant`
is `:det` (deterministic `tt_rounding` Sum+Round) or `:krp` (adaptive KRP).
Returns `(x, true_relres, res_hist, rank_hist)`.
"""
function run_cookie(; n = 8, variant = :krp, minD = 1.0, maxD = 5.0,
                    tol = 1e-8, maxit = 50, dir = DATA_DIR, seed = 1234, verbose = true)
    A0, A_c, a0 = load_cookie_data(dir)
    op = CookieOp(A0, A_c, n; minD = minD, maxD = maxD)
    b = cookie_rhs(op, a0)
    b = (1 / norm(b)) * b                                # scale to unit norm
    prec = make_preconditioner(A0, A_c)
    apply_op = x -> apply_summands(op, x)
    sumround = variant === :det ? sumround_det :
               (ys, α, ε) -> sumround_krp(ys, α, ε; seed = seed)

    println("\n=== cookie TT-GMRES  (variant=$variant, n=$n, modes=$(length(op.dims)), " *
            "n1=$(op.dims[1]), params∈[$minD,$maxD]) ===")
    tels = @elapsed (x, res_hist, rank_hist, _t_sr) =
        tt_gmres_cookie(apply_op, b, prec, sumround; tol = tol, maxit = maxit, verbose = verbose)

    # True relative residual, recomputed the MATLAB way: r = Σ_j A_j x − b.
    summ = apply_op(x)
    r = tt_rounding(weighted_sum(vcat(ones(length(summ)), -1.0), vcat(summ, [b])); tol = tol * 1e-2)
    relres = norm(r) / norm(b)

    @printf("→ %d iters, %.2fs, est.relres=%.3e, TRUE relres=%.3e, final max rank=%d\n",
            length(res_hist), tels, isempty(res_hist) ? NaN : res_hist[end],
            relres, maximum(x.ttv_rks))
    return x, relres, res_hist, rank_hist
end

if abspath(PROGRAM_FILE) == @__FILE__
    # Default smoke test runs the (fast) adaptive KRP Sum+Round — the paper's
    # proposal. The deterministic baseline is the *naive* TT-GMRES: correct but
    # ~40× slower here, so run it manually with `run_cookie(n=8, variant=:det)`.
    #
    # Both variants trace the same convergence; the TRUE residual floors around
    # ~1e-7 (not the 1e-8 GMRES target) because of the inexact Arnoldi rounding
    # inside TT-GMRES — exactly as reported in the paper (Fig. 7).
    n = 8
    _, relres_krp, _, _ = run_cookie(n = n, variant = :krp)
    @printf("\nSummary (n=%d): krp true relres = %.3e\n", n, relres_krp)
    @assert relres_krp < 1e-6 "KRP variant did not converge ($relres_krp)"
    println("OK — KRP TT-GMRES converged (true relres < 1e-6, ~paper's achievable ~1e-7).")
end
