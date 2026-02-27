"""
LiH ground state via randomized Lanczos / Rayleigh-Ritz in tensor-train format.

Reference: Tropp & Nakatsukasa, "Fast & Accurate Randomized Algorithms for
Linear Systems and Eigenvalue Problems" (2022).

FCIDUMP: 19 spatial orbitals, 4 electrons (2 alpha, 2 beta).
Active-space restriction to n_cas orbitals is supported.
"""

using TensorTrains
using LinearAlgebra
using JSON3
using TensorOperations

# ── Parameters ────────────────────────────────────────────────────────────────

const FCIDUMP = "./examples/lih_fcidump.txt"
n_cas    = 10     # number of active spatial orbitals  (≤ 19)
tol_mpo  = 1e-8   # SVD tolerance for MPO construction

# ── MPO cache ─────────────────────────────────────────────────────────────────

"""
    load_or_build_mpo(fcidump, n_cas, tol_mpo) -> H, E_nuc, n_elec

Return the LiH MPO for the given active space, loading from disk if a matching
cache file exists, otherwise building from scratch and saving to disk.
Cache files are named `lih_mpo_ncas<n>_tol<t>.json` in the same directory
as the FCIDUMP.
"""
function load_or_build_mpo(fcidump::String, n_cas::Int, tol_mpo::Float64)
    cache_dir  = dirname(abspath(fcidump))
    tol_str    = replace(string(tol_mpo), "-" => "m")   # e.g. "1em8"
    cache_file = joinpath(cache_dir, "lih_mpo_ncas$(n_cas)_tol$(tol_str).json")

    F = read_electron_integral_tensors(fcidump)
    E_nuc, n_elec = F[1], F[3]
    int_1e, int_2e = F[4], F[5]

    if isfile(cache_file)
        println("Loading MPO from cache: $cache_file")
        t = @elapsed H = json_to_mpo(JSON3.read(read(cache_file, String)))
        println("  loaded in $(round(t, digits=2))s")
        return H, E_nuc, n_elec
    end

    println("Building MPO (n_cas=$n_cas, tol=$tol_mpo) ...")
    h, V = one_e_two_e_integrals_to_hV(
        int_1e[1:n_cas, 1:n_cas],
        int_2e[1:n_cas, 1:n_cas, 1:n_cas, 1:n_cas],
    )
    t = @elapsed H = hV_to_mpo(h, V, ntuple(_->2, 2n_cas); tol=tol_mpo, chemistry=true)
    println("  built in $(round(t, digits=2))s  →  ranks: $(H.tto_rks)")

    println("Saving to cache: $cache_file")
    open(cache_file, "w") do io
        JSON3.pretty(io, H)
    end

    return H, E_nuc, n_elec
end

# ── Hamiltonian assembly ──────────────────────────────────────────────────────

H, E_nuc, n_elec = load_or_build_mpo(FCIDUMP, n_cas, tol_mpo)

# Reference Slater determinant (first n_elec spin-orbitals occupied)
ψ0 = slater(n_elec, 2n_cas)

# Sanity check: Hartree-Fock energy
E_hf = dot(ψ0, H * ψ0) + E_nuc
println("HF energy (active space, n_cas=$n_cas): $E_hf")
# Reference DMRG ground state: ≈ -7.9836 Ha (full active space)

# ── Sketched Arnoldi / Rayleigh-Ritz ─────────────────────────────────────────

"""
    expand_basis!(H, B_window, W_B_window, sketch_b, rmax; seed, k_trunc)
    -> b_new, sketch_b_new, sketch_Hb_prev

Add one vector to a sketch-orthogonal Krylov basis, reusing all precomputed sketch data.
Mutates `B_window` and `W_B_window` (appends b_new and its sketch, drops oldest if needed).

State passed in / mutated:
- `B_window`    : last k_trunc TTvectors (needed to propagate left contractions)
- `W_B_window`  : full recursive sketches of B_window (Vector{Matrix} per vector)
- `sketch_b`    : boundary sketch vectors Ω bₗ ∈ ℝˢ for all past l

The sketch of H*B_window[end] is computed at the start of the call when needed.
W_HB (full matrix) is discarded on return; its boundary `sketch_Hb_prev` (= Ω H b_{j-1})
is returned for the caller to store as `sketch_Hb[j-1]`.
"""
function expand_basis!(H::TToperator{T,N},
                      B_window::Vector{TTvector{T,N}},
                      W_B_window::Vector{Vector{Matrix{T}}},
                      sketch_b::Vector{Vector{T}},
                      rmax::Int;
                      seed::Int=1234,
                      k_trunc::Int=length(B_window)) where {T,N}

    j_new    = length(sketch_b) + 1
    l_range  = max(1, j_new - k_trunc):(j_new - 1)
    n_win    = length(B_window)
    b_prev   = B_window[end]
    dims     = b_prev.ttv_dims
    rks      = ones(Int, N+1); rks[1] = length(sketch_b[1]); rks[2:N] .= rmax

    # ── Sketch of H*b_prev
    WHb_raw, sk    = tt_recursive_sketch(T, H, b_prev, rks; orthogonal=true, reverse=true, seed=seed, block_rks=8)
    W_HB           = [reshape(WHb_raw[k], b_prev.ttv_rks[k]*H.tto_rks[k], sk[k]) for k=1:N+1]
    s_rks          = sk
    sketch_Hb_prev = vec(WHb_raw[1])   # boundary: Ω H b_{j-1}

    # ── Sketched GS coefficients (O(s · k_trunc), no TT ops) ──────────────
    h    = hcat(sketch_b[l_range]...) \ sketch_Hb_prev
    m    = 1 + length(l_range)          # terms: H b_{j-1}, then each b_l
    α_gs = T[one(T); -h...]

    # ortho_sketch = sketch_Hb_prev
    # for l in l_range
    #     ortho_sketch -= h[l-l_range[1]+1] * sketch_b[l]
    # end
    # j_new == 3 && @show dot(sketch_b[1], sketch_b[2])
    # @show j_new, [dot(ortho_sketch, sketch_b[l]) for l in l_range]

    # Helper: index of l in B_window / W_B_window
    win_idx(l) = n_win - (j_new - 1 - l)

    # ── QR sweep (left-to-right, using precomputed W matrices) ────────────
    vec_out = Vector{Array{T,3}}(undef, N)
    out_rks = ones(Int, N+1)

    # Initial left contractions at site 1 (left ranks are trivially 1; squeeze them)
    b1 = reshape(b_prev.ttv_vec[1], dims[1], b_prev.ttv_rks[2])
    H1 = reshape(H.tto_vec[1], dims[1], dims[1], H.tto_rks[2])
    @tensor Ay₁_tmp[i₁,α₂,β₂] := H1[i₁,j₁,β₂] * b1[j₁,α₂]
    Ay₁_tmp .*= α_gs[1]
    Ay₁ = reshape(Ay₁_tmp, dims[1], 1, b_prev.ttv_rks[2], H.tto_rks[2])

    Yₖ    = Vector{Array{T,3}}(undef, m)
    Yₖ[1] = reshape(Ay₁, dims[1], 1, b_prev.ttv_rks[2] * H.tto_rks[2])
    for (i, l) in enumerate(l_range)
        b_l   = B_window[win_idx(l)]
        Yₖ[1+i] = α_gs[1+i] .* reshape(b_l.ttv_vec[1], dims[1], 1, b_l.ttv_rks[2])
    end

    for k = 1:N-1
        # Sketch contraction: Zₖ accumulates all m terms
        Zₖ = zeros(T, dims[k], out_rks[k], s_rks[k+1])
        @tensoropt (αₖ₊₁,ρₖ,ρₖ₊₁) Zₖ[iₖ,ρₖ,ρₖ₊₁] +=
            Yₖ[1][iₖ,ρₖ,αₖ₊₁] * W_HB[k+1][αₖ₊₁,ρₖ₊₁]
        for (i, l) in enumerate(l_range)
            @tensoropt (αₖ₊₁,ρₖ,ρₖ₊₁) Zₖ[iₖ,ρₖ,ρₖ₊₁] +=
                Yₖ[1+i][iₖ,ρₖ,αₖ₊₁] * W_B_window[win_idx(l)][k+1][αₖ₊₁,ρₖ₊₁]
        end

        Q, _ = qr!(reshape(Zₖ, dims[k]*out_rks[k], s_rks[k+1]))
        Q    = Matrix(Q)
        out_rks[k+1] = size(Q, 2)
        vec_out[k]   = reshape(Q, dims[k], out_rks[k], out_rks[k+1])

        # Propagate left contractions to site k+1
        Yₖ₊₁ = Vector{Array{T,3}}(undef, m)

        y₁ₖ    = reshape(Yₖ[1], dims[k], out_rks[k], b_prev.ttv_rks[k+1], H.tto_rks[k+1])
        Ay₁ₖ₊₁ = zeros(T, dims[k+1], out_rks[k+1], b_prev.ttv_rks[k+2], H.tto_rks[k+2])
        @tensoropt (ρₖ,ρₖ₊₁,αₖ₊₁,βₖ₊₁,αₖ₊₂,βₖ₊₂) Ay₁ₖ₊₁[iₖ₊₁,ρₖ₊₁,αₖ₊₂,βₖ₊₂] =
            y₁ₖ[iₖ,ρₖ,αₖ₊₁,βₖ₊₁] * vec_out[k][iₖ,ρₖ,ρₖ₊₁] *
            b_prev.ttv_vec[k+1][jₖ₊₁,αₖ₊₁,αₖ₊₂] * H.tto_vec[k+1][iₖ₊₁,jₖ₊₁,βₖ₊₁,βₖ₊₂]
        Yₖ₊₁[1] = reshape(Ay₁ₖ₊₁, dims[k+1], out_rks[k+1],
                           b_prev.ttv_rks[k+2] * H.tto_rks[k+2])

        for (i, l) in enumerate(l_range)
            b_l = B_window[win_idx(l)]
            Yₖ₊₁[1+i] = zeros(T, dims[k+1], out_rks[k+1], b_l.ttv_rks[k+2])
            @tensoropt (αₖ₊₁,αₖ₊₂,ρₖ,ρₖ₊₁) Yₖ₊₁[1+i][iₖ₊₁,ρₖ₊₁,αₖ₊₂] =
                Yₖ[1+i][iₖ,ρₖ,αₖ₊₁] * vec_out[k][iₖ,ρₖ,ρₖ₊₁] *
                b_l.ttv_vec[k+1][iₖ₊₁,αₖ₊₁,αₖ₊₂]
        end

        Yₖ = Yₖ₊₁
    end

    # Last core: direct sum of all left contractions (all shape (dims[N], out_rks[N], 1))
    vec_out[N] = reshape(sum(Yₖ), dims[N], out_rks[N], 1)
    ot   = zeros(Int, N); ot[1:N-1] .= 1   # cores 1..N-1 are left-orthogonal; N is center
    b_new    = tt_rounding(TTvector{T,N}(N, vec_out, dims, out_rks, ot); rmax=rmax)

    # ── W_B_new: full sketch of b_new (enters the sliding window) ─────────
    W_B_new, _   = tt_recursive_sketch(T, b_new, rks; orthogonal=true, reverse=true, seed=seed, block_rks=8)
    
    β = norm(W_B_new[1])
    for i = 1:N # up to center of orthogonality of b_new
        W_B_new[i] ./= β
    end
    b_new /= β
    sketch_b_new = vec(W_B_new[1])

    push!(B_window,   b_new);    length(B_window)   > k_trunc && popfirst!(B_window)
    push!(W_B_window, W_B_new);  length(W_B_window) > k_trunc && popfirst!(W_B_window)

    return b_new, sketch_b_new, sketch_Hb_prev
end

"""
    sketched_arnoldi(H, b0, d, rmax; seed, k_trunc) -> B, C, D

Build a d-dimensional sketch-orthogonal Krylov basis for H starting from b0.
Uses `expand_basis!` to reuse sketch data across steps:
- W_HB is computed at the start of each expand_basis call and discarded (never stored between steps)
- W_B is kept in a sliding window of k_trunc entries

Returns:
- `B`: all d TTvectors
- `C`: s×d matrix, C[:,j] ≈ Ω bⱼ
- `D`: s×d matrix, D[:,j] ≈ Ω H bⱼ
"""
function sketched_arnoldi(H::TToperator{T,N}, b0::TTvector{T,N}, d::Int, rmax::Int;
                           seed::Int=1234, k_trunc::Int=d-1) where {T,N}
    rks = ones(Int, N+1); rks[1] = max(4d,rmax); rks[2:N] .= rmax

    B         = Vector{TTvector{T,N}}(undef, d)
    sketch_b  = Vector{Vector{T}}(undef, d)
    sketch_Hb = Vector{Vector{T}}(undef, d)

    # ── b₁ ──────────────────────────────────────────────────────────────────
    W_B_1, _    = tt_recursive_sketch(T, b0, rks; orthogonal=true, reverse=true, seed=seed, block_rks=8)

    β = norm(W_B_1[1])
    for i = 1:findfirst(isequal(0), b0.ttv_ot) # up to center of orthogonality of b0
        W_B_1[i] ./= β
    end
    B[1] = b0 / β
    sketch_b[1] = vec(W_B_1[1])

    B_window   = TTvector{T,N}[B[1]]
    W_B_window = [W_B_1]

    # ── Arnoldi iterations ───────────────────────────────────────────────────
    for j = 2:d
        b_new, sketch_b_j, sketch_Hb_prev =
            expand_basis!(H, B_window, W_B_window, sketch_b[1:j-1],
                          rmax; seed=seed, k_trunc=k_trunc)
        B[j]           = b_new
        sketch_b[j]    = sketch_b_j
        sketch_Hb[j-1] = sketch_Hb_prev   # sketch of H*b_{j-1}
    end

    # sketch of H*b_d (not produced by any expand_basis call)
    WHb_raw, _ = tt_recursive_sketch(T, H, B[d], rks; orthogonal=true, reverse=true, seed=seed, block_rks=8)
    sketch_Hb[d] = vec(WHb_raw[1])

    s = length(sketch_b[1])
    C = Matrix{T}(undef, s, d)
    D = Matrix{T}(undef, s, d)
    for j = 1:d
        C[:, j] = sketch_b[j]
        D[:, j] = sketch_Hb[j]
    end
    @show C'*C
    return B, C, D
end

"""
    sketched_rr(B, C, D, rmax; seed, κ_max) -> λs, rq, X

Stable sketched Rayleigh-Ritz (Tropp & Nakatsukasa 2022).

Given the Krylov basis B and sketch matrices C = Ω B, D = Ω H B (both s×d):
1. Thin SVD of C to check the condition number κ(C).
2. Truncate to the k ≤ d directions where σⱼ/σ₁ > 1/κ_max (basis whitening).
3. In the whitened coordinates, form the k×k projected Hamiltonian M = sym(Uᵀ D_white)
   and solve a standard (not generalized) eigenproblem.
4. Map eigenvectors back to the original B-basis and assemble TT Ritz vectors.

Returns k ≤ d Ritz values, the Rayleigh quotient of the ground-state Ritz vector,
and the k Ritz vectors.
"""
function sketched_rr(B::Vector{TTvector{T,N}}, C::Matrix{T}, D::Matrix{T},
                     rmax::Int; seed::Int=1234, κ_max::Real=1/sqrt(eps(T))) where {T,N}
    d = length(B)
    rks = ones(Int, N+1)
    rks[2:N] .= rmax

    # Thin SVD of C: s×d = U Σ Vᵀ
    F  = svd(C)
    σ  = F.S                    # length d, descending
    κ  = σ[1] / σ[end]
    println("Sketch condition number κ(C) ≈ $(round(κ, sigdigits=4))")

    # Truncate ill-conditioned directions
    k  = something(findlast(σ ./ σ[1] .> 1 / κ_max), 1)
    k < d && @warn "Basis truncated from d=$d to k=$k (ill-conditioned directions removed)"

    U  = F.U[:, 1:k]            # s×k, orthonormal columns
    Vk = F.V[:, 1:k]            # d×k
    σk = σ[1:k]

    # Projected Hamiltonian in orthonormal sketched basis (generalized eigenproblem)
    M  = U' * D * Vk           # k×k
    λs, Z = eigen(M, diagm(σk))   # real, sorted ascending

    # Ritz coefficients in original B-basis: Y = Vk Z  (d×k)
    Y = real.(Vk * Z)

    # Assemble Ritz vectors as TT linear combinations
    X = Vector{TTvector{T,N}}(undef, k)
    for i = 1:k
        X[i] = ttrand_rounding(Y[:, i], collect(B), rks; orthogonal=true, seed=seed)
    end
    return real.(λs), dot_operator(X[1], H, X[1]), X
end

"""
    dot_operator(ψ, H, φ) -> <ψ|H|φ>

Compute the bilinear form <ψ|H|φ> via a left-to-right transfer matrix sweep,
without forming the (potentially huge) product H*φ.
Transfer matrix at each step has size r_ψ × r_H × r_φ.
"""
function dot_operator(ψ::TTvector{T,N}, H::TToperator{T,N}, φ::TTvector{T,N}) where {T,N}
    L = ones(T, 1, 1, 1)
    for k = 1:N
        A_k = ψ.ttv_vec[k]   # (dims[k], r_ψ_left, r_ψ_right)
        H_k = H.tto_vec[k]   # (dims[k], dims[k], r_H_left, r_H_right)
        B_k = φ.ttv_vec[k]   # (dims[k], r_φ_left, r_φ_right)
        @tensoropt L[α,β,γ] := L[α_prev,β_prev,γ_prev] *
            conj(A_k[i,α_prev,α]) * H_k[i,j,β_prev,β] * B_k[j,γ_prev,γ]
    end
    return L[1,1,1]
end

"""
    exact_rr(H, B) -> eigenvalues sorted ascending

Exact Rayleigh-Ritz in the subspace spanned by B:
compute H_mat[i,j] = <B[i]|H|B[j]> and S_mat[i,j] = <B[i]|B[j]> via
transfer-matrix contractions (no explicit H*B[j] formed), then solve the
generalized eigenproblem.
"""
function exact_rr(H::TToperator{T,N}, B::Vector{TTvector{T,N}}) where {T,N}
    d     = length(B)
    H_mat = zeros(T, d, d)
    S_mat = zeros(T, d, d)
    for i = 1:d
        for j = i:d
            H_mat[i,j] = dot_operator(B[i], H, B[j])
            S_mat[i,j] = dot(B[i], B[j])
            H_mat[j,i] = H_mat[i,j]
            S_mat[j,i] = S_mat[i,j]
        end
    end
    return sort(eigvals(Symmetric(H_mat), Symmetric(S_mat)))
end

# ── Driver ────────────────────────────────────────────────────────────────────

d      = 30     # Krylov basis dimension
rmax   = 20     # max TT rank for sketching and Ritz vectors
seed   = 1234

B, C, D = sketched_arnoldi(H, ψ0, d, rmax; seed=seed, k_trunc=10)
λs, X   = sketched_rr(B, C, D, rmax; seed=seed)

println("Sketched Ritz values:      ", λs .+ E_nuc)
println("Sketched ground state:     ", λs[1] + E_nuc)

λs_exact = exact_rr(H, B)
println("Exact RR values:           ", λs_exact .+ E_nuc)
println("Exact RR ground state:     ", λs_exact[1] + E_nuc, "  (ref ≈ -7.9836 Ha)")

# ── DMRG reference at rmax ─────────────────────────────────────────────────
v_dmrg = tt_up_rks(ψ0, 16; ϵ_wn=1e-2)
schedule = dmrg_schedule_default(; rmax=rmax)
E_hist, ψ_dmrg, _ = dmrg_eigsolv(H, v_dmrg; schedule=schedule, verbose=false)
println("DMRG ground state (rmax=$rmax): ", E_hist[end] + E_nuc)
