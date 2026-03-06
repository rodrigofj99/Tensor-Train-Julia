"""
Transcorrelated Be ground state via randomized Lanczos / Rayleigh-Ritz
in tensor-train format.

Reference: Tropp & Nakatsukasa, "Fast & Accurate Randomized Algorithms for
Linear Systems and Eigenvalue Problems" (2022).

Hamiltonian: transcorrelated Be integrals from Be.ezfio.FCIDUMP (full active space).
Reference energy (Giner): -14.6251920 Ha.
"""

using TensorTrains
using LinearAlgebra
using JSON3
using TensorOperations
using CairoMakie


# ── Parameters ────────────────────────────────────────────────────────────────

const FCIDUMP = "./examples/Be.ezfio.FCIDUMP"
const E_REF   = -14.6251920    # Giner reference energy (Ha)
tol_mpo = 1e-10                # SVD tolerance for MPO construction


# ── MPO cache ─────────────────────────────────────────────────────────────────

"""
    load_or_build_mpo(fcidump, tol_mpo) -> H, E_nuc, n_elec, n_orbs

Return the Be transcorrelated MPO, loading from disk if a matching cache file
exists, otherwise building from scratch and saving to disk.
Cache files are named `<fcidump>_mpo_tol<t>.json` in the same directory.
"""
function load_or_build_mpo(fcidump::String, tol_mpo::Float64=1e-10)
    cache_dir  = dirname(abspath(fcidump))
    name       = basename(fcidump)
    tol_str    = replace(string(tol_mpo), "-" => "m")
    cache_file = joinpath(cache_dir, "$(name)_mpo_tol$(tol_str).json")

    F = read_electron_integral_tensors_nosymmetry(fcidump)
    E_nuc, n_elec, n_orbs = F[1], F[3], F[2]
    int_1e, int_2e = F[4], F[5]

    if isfile(cache_file)
        println("Loading MPO from cache: $cache_file")
        t = @elapsed H = json_to_mpo(JSON3.read(read(cache_file, String)))
        println("  loaded in $(round(t, digits=2))s")
        return H, E_nuc, n_elec, n_orbs
    end

    println("Building MPO (tol=$tol_mpo) ...")
    h, V = one_e_two_e_integrals_to_hV(int_1e, int_2e)
    t = @elapsed H = TensorTrains.hV_no_to_mpo(h, V, ntuple(_->2, 2n_orbs); tol=tol_mpo)
    println("  built in $(round(t, digits=2))s  →  ranks: $(H.tto_rks)")

    println("Saving to cache: $cache_file")
    open(cache_file, "w") do io
        JSON3.pretty(io, H)
    end

    return H, E_nuc, n_elec, n_orbs
end

# ── Hamiltonian assembly ──────────────────────────────────────────────────────

H, E_nuc, n_elec, n_orbs = load_or_build_mpo(FCIDUMP, tol_mpo)

# Reference Slater determinant (first n_elec spin-orbitals occupied)
ψ0 = slater(n_elec, 2n_orbs)

# Particle number operator
N = part_num(ψ0.ttv_dims)

# Sanity check: HF-like energy (Rayleigh quotient of ψ0)
E_hf = real(dot(ψ0, H * ψ0)) + E_nuc
println("HF-like energy: $E_hf  (Giner ref: $E_REF)")

# ── Algorithm implementations ─────────────────────────────────────────────────

"""
    expand_basis!(H, B_window, W_B_window, B_sketch_window, rks; ...) -> b_new, sketch_b_new, sketch_Hb_prev

Add one vector to a sketch-orthogonal Krylov basis, reusing all precomputed sketch data.
Mutates `B_window`, `W_B_window`, and `B_sketch_window` (appends `b_new` and its sketch,
drops the oldest entry if the sliding window exceeds `k_trunc`).

The sketch of `H*B_window[end]` is computed at the start of the call.
Its boundary vector `sketch_Hb_prev = Ω H b_{j-1}` is returned for the caller to store.

# Arguments
- `H`: TT operator
- `B_window`: sliding window of the last `k_trunc` basis TTvectors
- `W_B_window`: full recursive sketches `{W_B[k]}` for each vector in `B_window`
- `B_sketch_window`: boundary sketch vectors `Ω bₗ ∈ ℝˢ` for each vector in `B_window`
- `rks`: sketch rank vector (length N+1)
- `k_trunc`: maximum window size (default: current window length)
"""
function expand_basis!(H::TToperator{T,N},
                      B_window::Vector{TTvector{T,N}},
                      W_B_window::Vector{Vector{Matrix{T}}},
                      B_sketch_window::Vector{Vector{T}},
                      rks::Vector{Int};
                      orthogonal=orthogonal,
                      block_rks=8,
                      seed::Int=1234,
                      k_trunc::Int=length(B_window)) where {T,N}

    b_prev   = B_window[end]
    dims     = b_prev.ttv_dims

    # ── Sketch of H*b_prev
    WHb_raw, sk    = tt_recursive_sketch(T, H, b_prev, rks; orthogonal=orthogonal, reverse=true, seed=seed, block_rks=block_rks)
    W_HB           = [reshape(WHb_raw[k], b_prev.ttv_rks[k]*H.tto_rks[k], sk[k]) for k=1:N+1]
    s_rks          = sk
    sketch_Hb_prev = vec(WHb_raw[1])   # boundary: Ω H b_{j-1}

    # ── Sketched GS coefficients (O(s · k_trunc), no TT ops) ──────────────
    h    = hcat(B_sketch_window...) \ sketch_Hb_prev
    m    = 1 + length(B_window)         # terms: H b_{j-1}, then each b_l

    # ── QR sweep (left-to-right, using precomputed W matrices) ────────────
    vec_out = Vector{Array{T,3}}(undef, N)
    out_rks = ones(Int, N+1)
    ot = zeros(Int, N)

    # Initial left contractions at site 1 (left ranks are trivially 1; squeeze them)
    b1 = reshape(b_prev.ttv_vec[1], dims[1], b_prev.ttv_rks[2])
    H1 = reshape(H.tto_vec[1], dims[1], dims[1], H.tto_rks[2])
    @tensor Ay₁_tmp[i₁,α₂,β₂] := H1[i₁,j₁,β₂] * b1[j₁,α₂]
    Ay₁ = reshape(Ay₁_tmp, dims[1], 1, b_prev.ttv_rks[2], H.tto_rks[2])

    Yₖ    = Vector{Array{T,3}}(undef, m)
    Yₖ[1] = reshape(Ay₁, dims[1], 1, b_prev.ttv_rks[2] * H.tto_rks[2])
    for (i,b) in enumerate(B_window)
        Yₖ[1+i] = -h[i] .* b.ttv_vec[1]
    end

    for k = 1:N-1
        # Sketch contraction: Zₖ accumulates all m terms
        Zₖ = zeros(T, dims[k], out_rks[k], s_rks[k+1])
        @tensoropt (αₖ₊₁,ρₖ,ρₖ₊₁) Zₖ[iₖ,ρₖ,ρₖ₊₁] += Yₖ[1][iₖ,ρₖ,αₖ₊₁] * W_HB[k+1][αₖ₊₁,ρₖ₊₁]
        for (i,W) in enumerate(W_B_window)
            @tensoropt (αₖ₊₁,ρₖ,ρₖ₊₁) Zₖ[iₖ,ρₖ,ρₖ₊₁] += Yₖ[1+i][iₖ,ρₖ,αₖ₊₁] * W[k+1][αₖ₊₁,ρₖ₊₁]
        end

        Q, _ = qr!(reshape(Zₖ, dims[k]*out_rks[k], s_rks[k+1]))
        Q    = Matrix(Q)
        out_rks[k+1] = size(Q, 2)
        vec_out[k]   = reshape(Q, dims[k], out_rks[k], out_rks[k+1])
        ot[k] = 1

        # Propagate left contractions to site k+1
        Yₖ₊₁ = Vector{Array{T,3}}(undef, m)

        y₁ₖ    = reshape(Yₖ[1], dims[k], out_rks[k], b_prev.ttv_rks[k+1], H.tto_rks[k+1])
        Ay₁ₖ₊₁ = zeros(T, dims[k+1], out_rks[k+1], b_prev.ttv_rks[k+2], H.tto_rks[k+2])
        @tensoropt (ρₖ,ρₖ₊₁,αₖ₊₁,βₖ₊₁,αₖ₊₂,βₖ₊₂) Ay₁ₖ₊₁[iₖ₊₁,ρₖ₊₁,αₖ₊₂,βₖ₊₂] = y₁ₖ[iₖ,ρₖ,αₖ₊₁,βₖ₊₁] * vec_out[k][iₖ,ρₖ,ρₖ₊₁] * b_prev.ttv_vec[k+1][jₖ₊₁,αₖ₊₁,αₖ₊₂] * H.tto_vec[k+1][iₖ₊₁,jₖ₊₁,βₖ₊₁,βₖ₊₂]
        Yₖ₊₁[1] = reshape(Ay₁ₖ₊₁, dims[k+1], out_rks[k+1],
                           b_prev.ttv_rks[k+2] * H.tto_rks[k+2])

        for (i,b) in enumerate(B_window)
            Yₖ₊₁[1+i] = zeros(T, dims[k+1], out_rks[k+1], b.ttv_rks[k+2])
            @tensoropt (αₖ₊₁,αₖ₊₂,ρₖ,ρₖ₊₁) Yₖ₊₁[1+i][iₖ₊₁,ρₖ₊₁,αₖ₊₂] =
                Yₖ[1+i][iₖ,ρₖ,αₖ₊₁] * vec_out[k][iₖ,ρₖ,ρₖ₊₁] *
                b.ttv_vec[k+1][iₖ₊₁,αₖ₊₁,αₖ₊₂]
        end

        Yₖ = Yₖ₊₁
    end

    # Last core: direct sum of all left contractions (all shape (dims[N], out_rks[N], 1))
    vec_out[N] = reshape(sum(Yₖ), dims[N], out_rks[N], 1)
    b_new = tt_rounding(TTvector{T,N}(N, vec_out, dims, out_rks, ot); rmax=maximum(rks[2:end]))

    # ── W_B_new: full sketch of b_new (enters the sliding window) ─────────
    W_B_new, _ = tt_recursive_sketch(T, b_new, rks; orthogonal=orthogonal, reverse=true, seed=seed, block_rks=block_rks)

    β = norm(W_B_new[1])
    for i = 1:findfirst(b_new.ttv_ot.==0)
        W_B_new[i] ./= β
    end
    b_new /= β
    sketch_b_new = vec(W_B_new[1])

    push!(B_window,        b_new);         length(B_window)        > k_trunc && popfirst!(B_window)
    push!(B_sketch_window, sketch_b_new);  length(B_sketch_window) > k_trunc && popfirst!(B_sketch_window)
    push!(W_B_window,      W_B_new);       length(W_B_window)      > k_trunc && popfirst!(W_B_window)

    return b_new, sketch_b_new, sketch_Hb_prev
end

"""
    sketched_rr(H, B, C, D, rmax; ...) -> λs, X, res_sketch, res_sample

Stable sketched Rayleigh-Ritz (Tropp & Nakatsukasa 2022).

Given the Krylov basis `B` and sketch matrices `C = Ω B`, `D = Ω H B` (both s×d):
- `stable=false` (default): column-pivoted QR of C followed by a standard eigenproblem
  on the projected operator `C \\ D`.
- `stable=true`: thin SVD of C for basis whitening, truncating ill-conditioned
  directions (threshold `κ_max`), then a generalized eigenproblem.

Returns Ritz values `λs`, Ritz vectors `X`, the sketched residual
`‖D y₁ − λ₁ C y₁‖`, and a sampled estimate of the true residual
`‖H x₁ − λ₁ x₁‖ / ‖x₁‖`.
"""
function sketched_rr(H::TToperator{T,N}, B::Vector{TTvector{T,N}}, C::Matrix{T}, D::Matrix{T},
                     rmax::Int; orthogonal=orthogonal, block_rks=8, seed::Int=1234,
                     κ_max::Real=1/sqrt(eps(real(T))), stable=false) where {T,N}
    d = length(B)

    if stable
        F  = svd(C)
        σ  = F.S
        println("Sketch condition number κ(C) ≈ $(round(σ[1]/σ[end], sigdigits=4))")
        k  = something(findlast(σ ./ σ[1] .> 1 / κ_max), 1)
        k < d && @warn "Basis truncated from d=$d to k=$k (ill-conditioned directions removed)"

        U  = F.U[:, 1:k]
        Vk = F.V[:, 1:k]
        σk = σ[1:k]
        M  = U' * D * Vk
        λs, Z = eigen(M, diagm(σk))
        Y = real.(Vk * Z)
        for j = 1:k
            Y[:,j] = Y[:,j] / norm(C*Y[:,j])
        end
    else
        k = d
        F = qr(C, ColumnNorm())
        M = F \ D
        λs, Y = eigen(M)
        Y = real.(Y)
    end

    λs = real.(λs)
    res_sketch = norm(D*Y[:,1] - λs[1]*C*Y[:,1])

    # Assemble Ritz vectors as TT linear combinations
    X = Vector{TTvector{T,N}}(undef, k)
    for i = 1:k
        y = Y[:, i] / norm(Y[:,i])
        X[i] = ttrand_rounding(y, collect(B), 4rmax; orthogonal=orthogonal, seed=seed, block_rks=block_rks)
        X[i] = tt_rounding(X[i]; tol=res_sketch^2/2)
        X[i] = X[i] / norm(X[i])
    end

    # Sampled estimate of the true residual ‖H x₁ − λ₁ x₁‖ / ‖x₁‖
    rks_samp  = ones(Int, N+1); rks_samp[1:N] .= N
    seed_samp = rand(Int)
    W1 = tt_recursive_sketch(T, H, X[1], rks_samp; orthogonal=orthogonal, reverse=true, seed=seed_samp, block_rks=block_rks)[1]
    W2 = tt_recursive_sketch(T,    X[1], rks_samp; orthogonal=orthogonal, reverse=true, seed=seed_samp, block_rks=block_rks)[1]
    res_sample = norm(vec(W1[1]) - λs[1]*vec(W2[1])) / norm(W2[1])

    return λs, X, res_sketch, res_sample
end

"""
    sketched_rayleigh_ritz(H, b0, d, rmax; ...) -> λ, ψ_rr, history

Build a `d`-dimensional sketch-orthogonal Krylov basis for `H` starting from `b0`,
interleaving sketched Rayleigh-Ritz (sRR) at each step to track convergence.

At each Arnoldi expansion step `j = 2, …, d`, the sketch `Ω H b_{j-1}` becomes
available and sRR is applied to the basis `B[1:j-1]`. A final sRR with all `d`
vectors is performed after the basis is complete.

# Arguments
- `H`: TT operator (Hamiltonian; may be non-Hermitian for transcorrelated systems)
- `b0`: initial TT vector (sketch-normalized internally)
- `d`: Krylov dimension
- `rmax`: max TT rank for sketch ranks and compressed Ritz vectors
- `orthogonal`: use orthogonal sketch matrices (default: global `orthogonal`)
- `block_rks`: sketch block size (default: 8)
- `seed`: random seed (default: 1234)
- `k_trunc`: sliding window size for sketch orthogonalization (default: `d-1`)
- `e_nuc`: nuclear repulsion energy added to all stored/returned eigenvalues (default: 0)

# Returns
- `λ`: lowest sketched Ritz value + `e_nuc`
- `ψ_rr`: ground-state Ritz vector as a TT
- `history`: `Vector` of `NamedTuple`, one entry per sRR solve, with fields:
  - `iter`: Krylov dimension `k` used in that sRR call (1 ≤ k ≤ d)
  - `e_sketch`: sketched Ritz estimate `λ₁ + e_nuc`
  - `e_true`: real part of Rayleigh quotient `Re⟨ψ|H|ψ⟩ + e_nuc`
  - `res_sketch`: sketched residual `‖D y₁ − λ₁ C y₁‖`
  - `res_sample`: sampled estimate of `‖H ψ − λ ψ‖ / ‖ψ‖`
"""
function sketched_rayleigh_ritz(H::TToperator{T,N}, b0::TTvector{T,N}, d::Int, rmax::Int;
                           orthogonal=orthogonal, block_rks=8, seed::Int=1234,
                           k_trunc::Int=d-1, e_nuc::Real=0.0) where {T,N}
    rks = ones(Int, N+1); rks[1] = max(4d, rmax); rks[2:N] .= rmax

    B         = Vector{TTvector{T,N}}(undef, d)
    sketch_b  = Vector{Vector{T}}(undef, d)
    sketch_Hb = Vector{Vector{T}}(undef, d)

    # ── b₁: sketch-normalize b0 ───────────────────────────────────────────────
    W_B_1, _ = tt_recursive_sketch(T, b0, rks; orthogonal=orthogonal, reverse=true, seed=seed, block_rks=block_rks)

    β = norm(W_B_1[1])
    for i = 1:findfirst(isequal(0), b0.ttv_ot)
        W_B_1[i] ./= β
    end
    B[1] = b0 / β
    sketch_b[1] = vec(W_B_1[1])
    s = length(sketch_b[1])

    B_window        = TTvector{T,N}[B[1]]
    W_B_window      = [W_B_1]
    B_sketch_window = [sketch_b[1]]

    history = NamedTuple[]

    # ── Arnoldi iterations with interleaved sRR ───────────────────────────────
    for j = 2:d
        b_new, sketch_b_j, sketch_Hb_prev =
            expand_basis!(H, B_window, W_B_window, B_sketch_window,
                          rks; block_rks=block_rks, seed=seed, k_trunc=k_trunc)
        B[j]           = b_new
        sketch_b[j]    = sketch_b_j
        sketch_Hb[j-1] = sketch_Hb_prev

        # sRR with B[1:j-1] (sketch of H*b_{j-1} just became available)
        C = reduce(hcat, sketch_b[1:j-1])
        D = reduce(hcat, sketch_Hb[1:j-1])
        λs, X, res_sketch, res_sample = sketched_rr(H, B[1:j-1], C, D, rmax;
                                                     block_rks=block_rks, seed=seed)
        ψ_rr     = X[1]
        e_sketch = λs[1] + e_nuc
        e_true   = real(dot_operator(ψ_rr, H, ψ_rr)) + e_nuc

        push!(history, (iter=j-1, e_sketch=e_sketch, e_true=e_true,
                        res_sketch=res_sketch, res_sample=res_sample))

        println("─── k = $(j-1) ────────────────────────────────────────────")
        println("  Sketched Ritz:    ", e_sketch)
        println("  True Ritz:        ", e_true)
        println("  Sketch residual:  ", res_sketch)
        println("  Sampled residual: ", res_sample)
    end

    # ── Final sRR with all d basis vectors ────────────────────────────────────
    WHb_raw, _ = tt_recursive_sketch(T, H, B[d], rks; orthogonal=orthogonal, reverse=true, seed=seed, block_rks=block_rks)
    sketch_Hb[d] = vec(WHb_raw[1])

    C = reduce(hcat, sketch_b)
    D = reduce(hcat, sketch_Hb)
    λs, X, res_sketch, res_sample = sketched_rr(H, B, C, D, rmax;
                                                 block_rks=block_rks, seed=seed)
    ψ_rr     = X[1]
    e_sketch = λs[1] + e_nuc
    e_true   = real(dot_operator(ψ_rr, H, ψ_rr)) + e_nuc

    push!(history, (iter=d, e_sketch=e_sketch, e_true=e_true,
                    res_sketch=res_sketch, res_sample=res_sample))

    println("─── k = $d (final) ─────────────────────────────────────────────")
    println("  Sketched Ritz:    ", e_sketch)
    println("  True Ritz:        ", e_true)
    println("  Sketch residual:  ", res_sketch)
    println("  Sampled residual: ", res_sample)

    return e_sketch, ψ_rr, history
end

"""
    plot_sRR_convergence(stages; e_ref, filename) -> Figure

Plot convergence diagnostics from one or more `sketched_rayleigh_ritz` histories
on a single figure with cumulative iteration counts.

`stages` is a vector of `(label, history)` pairs, or a single history vector
(treated as one unlabeled stage). Each stage is drawn in a distinct color;
iteration counts are offset so stages appear consecutively on the x-axis.

Two panels: (left) |E − Eref| vs iteration count (log10 y-scale);
(right) sketched and sampled residuals (log10 y-scale).
If `e_ref` is `nothing`, raw energies are shown on a linear scale.

# Arguments
- `stages`: `Vector{Tuple{String, Vector{NamedTuple}}}` or a single history
- `e_ref`: reference energy (optional)
- `filename`: save path (format inferred from extension)
"""
function plot_sRR_convergence(stages; e_ref=nothing, dmrg_refs=nothing, filename=nothing)
    # Accept a bare history as a single unnamed stage
    if stages isa Vector{<:NamedTuple}
        stages = [("", stages)]
    end

    n      = length(stages)
    _cg    = cgrad(:plasma)
    _cmap  = cgrad([_cg[t] for t in range(0.0, 0.85, length=256)])
    colors = [_cg[t] for t in range(0.0, 0.85, length=n)]

    theme = Theme(
        fontsize  = 12,
        font      = "Computer Modern",
        linewidth = 2,
        Axis = (
            titlesize          = 12,
            xlabelsize         = 11,
            ylabelsize         = 11,
            xticklabelsize     = 10,
            yticklabelsize     = 10,
            spinewidth         = 1.5,
            xtickwidth         = 1.5,
            ytickwidth         = 1.5,
            xgridwidth         = 0.75,
            ygridwidth         = 0.75,
            xgridcolor         = :gray90,
            ygridcolor         = :gray90,
            xgridvisible       = true,
            ygridvisible       = true,
            topspinevisible    = true,
            rightspinevisible  = true,
            leftspinecolor     = :black,
            rightspinecolor    = :black,
            topspinecolor      = :black,
            bottomspinecolor   = :black,
        ),
        Legend = (
            labelsize       = 11,
            titlesize       = 11,
            framevisible    = true,
            backgroundcolor = (:white, 0.9),
            framecolor      = :black,
            framewidth      = 1,
        ),
    )

    with_theme(theme) do
        ylabel_left = isnothing(e_ref) ? L"$E$ (Ha)" :
                      L"$\langle \psi | H | \psi \rangle - E_\mathrm{ref}$ (Ha)"
        title_left  = isnothing(e_ref) ? L"\text{Ground state energy}" : L"\text{Ground state energy error}"
        yscale_left = isnothing(e_ref) ? identity : log10

        fig = Figure(size=(624, 380))

        ax1 = Axis(fig[1, 1], xlabel=L"\text{Iteration count}",
                   ylabel=ylabel_left, title=title_left, yscale=yscale_left)
        ax2 = Axis(fig[1, 2], xlabel=L"\text{Iteration count}",
                   ylabel=L"Estimated $\Vert H\psi - \lambda\psi \Vert^2$",
                   title=L"\text{Residual convergence}", yscale=log10)

        offset = 0
        for (i, (label, history)) in enumerate(stages)
            iters       = [h.iter + offset for h in history]
            vals_sketch = [h.e_sketch      for h in history]
            vals_true   = [h.e_true        for h in history]
            res_sketch  = [h.res_sketch    for h in history]
            res_sample  = [h.res_sample    for h in history]

            c = colors[i]

            if isnothing(e_ref)
                lines!(ax1, iters, vals_true,          color=c)
                scatterlines!(ax1, iters, vals_sketch, color=c, linestyle=:dash)
            else
                nan_pos(v) = v > 0 ? v : NaN
                lines!(ax1,        iters, nan_pos.(vals_true   .- e_ref), color=c)
                scatterlines!(ax1, iters, abs.(vals_sketch .- e_ref),     color=c, linestyle=:dash)
            end
            lines!(ax2,        iters, res_sample.^2, color=c)
            scatterlines!(ax2, iters, res_sketch.^2, color=c, linestyle=:dash)

            # DMRG reference line for this stage (left panel only)
            if !isnothing(dmrg_refs) && i <= length(dmrg_refs) && !isnothing(dmrg_refs[i])
                val = isnothing(e_ref) ? dmrg_refs[i] : dmrg_refs[i] - e_ref
                val > 0 && lines!(ax1, [iters[1], iters[end]], [val, val],
                                  color=c, linestyle=:dot, linewidth=1.5)
            end

            offset = iters[end]
        end

        # Line-type legend: solid = estimated/true, dashed+markers = sketched
        style_elems  = [LineElement(color=:black, linewidth=2, linestyle=:solid),
                        LineElement(color=:black, linewidth=2, linestyle=:dash)]
        style_labels = [L"\text{Estimated / True}", L"\text{Sketched}"]
        if !isnothing(dmrg_refs) && any(!isnothing, dmrg_refs)
            push!(style_elems,  LineElement(color=:black, linewidth=1.5, linestyle=:dot))
            push!(style_labels, L"\text{DMRG bound}")
        end

        Legend(fig[2, 1], [style_elems], [style_labels], [L"\text{Line type}"],
               orientation=:horizontal, tellwidth=false)

        # Colorbar: color encodes rank
        rank_labels = [label for (label, _) in stages]
        rank_ticks  = LaTeXString.("\$" .* chopprefix.(rank_labels, "rmax=") .* "\$")
        Colorbar(fig[2, 2], colormap=_cmap,
                 limits=(0.5, n + 0.5),
                 ticks=(collect(1:n), rank_ticks),
                 label=L"\text{Rank}",
                 vertical=false,
                 flipaxis=false,
                 tellwidth=false)

        isnothing(filename) || save(filename, fig)
        return fig
    end
end

"""
    sketched_rayleigh_ritz(H, b0, schedule; ...) -> λ, ψ_rr, stages, seeds_used

Multi-stage sRR for non-Hermitian (transcorrelated) operators.
Schedule entries require `d` and `rmax`; optional: `seed`, `k_trunc`, `label`.
"""
function sketched_rayleigh_ritz(H::TToperator{T,N}, b0::TTvector{T,N},
                                  schedule::Vector{<:NamedTuple};
                                  orthogonal::Bool=true, block_rks::Int=8,
                                  seed::Int=1234, e_nuc::Real=0.0,
                                  validate_seed::Bool=false) where {T,N}
    stages     = Tuple{String, Vector{NamedTuple}}[]
    seeds_used = Vector{Int}(undef, length(schedule))
    ψ_rr       = b0
    λ          = 0.0

    for (i, stage) in enumerate(schedule)
        d    = stage.d
        rmax = stage.rmax
        s    = get(stage, :seed,    seed + i - 1)
        kt   = get(stage, :k_trunc, d - 1)
        lbl  = get(stage, :label,   "d=$d, rmax=$rmax")

        b_init = i == 1 ? ψ_rr : tt_rounding(ψ_rr; rmax=rmax)

        if validate_seed
            rks_check = vcat(max(4d, rmax), fill(rmax, N-1), 1)
            while abs(norm(tt_recursive_sketch(T, b_init, rks_check;
                           orthogonal=orthogonal, reverse=true,
                           seed=s, block_rks=block_rks)[1][1])^2 - 1) > 0.1
                s = rand(Int)
            end
            println("Stage $i: using seed $s")
        end

        seeds_used[i] = s
        λ, ψ_rr, hist = sketched_rayleigh_ritz(H, b_init, d, rmax;
                                                orthogonal=orthogonal,
                                                block_rks=block_rks,
                                                seed=s, k_trunc=kt,
                                                e_nuc=e_nuc)
        push!(stages, (lbl, hist))
    end

    return λ, ψ_rr, stages, seeds_used
end

"""
    save_sRR_results(filename, stages, seeds, schedule, λ)

Serialize sRR output to a JSON file.
"""
function save_sRR_results(filename::String,
                          stages,
                          seeds::Vector{Int},
                          schedule::Vector{<:NamedTuple},
                          λ::Real)
    data = (
        final_energy = λ,
        seeds        = seeds,
        schedule     = schedule,
        stages       = [(label=lbl, history=hist) for (lbl, hist) in stages],
    )
    open(filename, "w") do io
        JSON3.pretty(io, data)
    end
    println("sRR results saved to $filename")
end

"""
    load_sRR_results(filename) -> stages, seeds, λ

Load sRR results previously written by `save_sRR_results`.
Returns `stages` in the format expected by `plot_sRR_convergence`.
"""
function load_sRR_results(filename::String)
    raw = JSON3.read(read(filename, String))
    stages = [
        (String(s.label),
         [(iter       = Int(h.iter),
           e_sketch   = Float64(h.e_sketch),
           e_true     = Float64(h.e_true),
           res_sketch = Float64(h.res_sketch),
           res_sample = Float64(h.res_sample))
          for h in s.history])
        for s in raw.stages
    ]
    seeds = Vector{Int}(raw.seeds)
    return stages, seeds, Float64(raw.final_energy)
end


# ── Driver ────────────────────────────────────────────────────────────────────

block_rks  = 8
orthogonal = true

schedule = [
    (d=20, rmax=10, k_trunc=20, label="rmax=10"),
    (d=20, rmax=20, k_trunc=20, label="rmax=20"),
]

e, ψ_rr, stages, seeds = sketched_rayleigh_ritz(H, ψ0, schedule;
                    orthogonal=orthogonal, block_rks=block_rks,
                    seed=rand(Int), e_nuc=E_nuc,
                    validate_seed=true)

println("Final energy:    ", e)
println("True Ritz:       ", real(dot_operator(ψ_rr, H, ψ_rr)) + E_nuc,
        ";  Error: ", real(dot_operator(ψ_rr, H, ψ_rr)) + E_nuc - E_REF)
println("Particle number: ", real(dot_operator(ψ_rr, N, ψ_rr)))

save_sRR_results("be_sRR.json", stages, seeds, schedule, e)

# ── Convergence plot ──────────────────────────────────────────────────────────
plot_sRR_convergence(stages; e_ref=E_REF, filename="convergence_Be.pdf")
