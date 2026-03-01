"""
LiH ground state via randomized Lanczos / Rayleigh-Ritz in tensor-train format.

Reference: Tropp & Nakatsukasa, "Fast & Accurate Randomized Algorithms for
Linear Systems and Eigenvalue Problems" (2022).

FCIDUMP: 19 spatial orbitals, 4 electrons (2 alpha, 2 beta).
Active-space restriction to n_cas orbitals is supported.
"""

using Revise
using PythonCall
using TensorTrains
using LinearAlgebra
using JSON3


# ── Parameters ────────────────────────────────────────────────────────────────

const FCIDUMP = "./examples/lih_fcidump.txt"
n_cas    = 19     # number of active spatial orbitals  (≤ 19)
tol_mpo  = 1e-8   # SVD tolerance for MPO construction



pyscf = pyimport("pyscf")
fci = pyimport("pyscf.fci")
cc = pyimport("pyscf.cc")
mcscf = pyimport("pyscf.mcscf")


mol = pyscf.gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="ccpvdz", unit="Angstrom", verbose = 3)
mf = pyscf.scf.RHF(mol).run()
println("RHF Energy (Ha): ", mf.e_tot)

myccsd = pyscf.cc.CCSD(mf,frozen=collect(n_cas:18)).run()
mycas = pyscf.mcscf.CASCI(mf, n_cas, 4).run()

e_cas = pyconvert(Float64, mycas.e_cas)
e_tot = pyconvert(Float64, mycas.e_tot)
E_nuc = pyconvert(Float64, mycas.energy_nuc())
e_ref = e_tot

@show e_cas, e_tot


# ── MPO cache ─────────────────────────────────────────────────────────────────

"""
    load_or_build_mpo(fcidump, n_cas, tol_mpo) -> H, E_nuc, n_elec

Return the LiH MPO for the given active space, loading from disk if a matching
cache file exists, otherwise building from scratch and saving to disk.
Cache files are named `lih_mpo_ncas<n>_tol<t>.json` in the same directory
as the FCIDUMP.
"""
function load_or_build_mpo(fcidump::String, n_cas::Int, tol_mpo::Float64, μ::Float64=0.2)
    cache_dir  = dirname(abspath(fcidump))
    tol_str    = replace(string(tol_mpo), "-" => "m")
    shift_str  = replace(string(μ), "-" => "m")
    cache_file = joinpath(cache_dir, "lih_mpo_ncas$(n_cas)_tol$(tol_str)_shift$(shift_str).json")

    F = read_electron_integral_tensors(fcidump)
    E_nuc, n_elec = F[1], F[3]
    int_1e, int_2e = F[4], F[5]
    v = slater(F[3],2n_cas)

    if isfile(cache_file)
        println("Loading MPO from cache: $cache_file")
        t = @elapsed H = json_to_mpo(JSON3.read(read(cache_file, String)))
        println("  loaded in $(round(t, digits=2))s")
        return H, E_nuc-n_elec*μ, n_elec
    end

    println("Building MPO (n_cas=$n_cas, tol=$tol_mpo) ...")
    h, V = one_e_two_e_integrals_to_hV(
        int_1e[1:n_cas, 1:n_cas],
        int_2e[1:n_cas, 1:n_cas, 1:n_cas, 1:n_cas],
    )
    t_tree = @elapsed H = hV_to_mpo_tree(h+μ*I, V, ntuple(_->2, 2n_cas); tol=tol_mpo, chemistry=true)
    println("  hV_to_mpo_tree: $(round(t_tree, digits=2))s  →  ranks: $(H.tto_rks)")
    # t_seq  = @elapsed H_seq = hV_to_mpo(h+μ*I, V, ntuple(_->2, 2n_cas); tol=tol_mpo, chemistry=true)
    # println("  hV_to_mpo (seq): $(round(t_seq, digits=2))s  →  ranks: $(H_seq.tto_rks)")
    # println("  speedup: $(round(t_seq/t_tree, digits=2))×")
    println(dot(v,H*v)+F[1]-F[3]*μ) # =-7.9836

    println("Saving to cache: $cache_file")
    open(cache_file, "w") do io
        JSON3.pretty(io, H)
    end

    return H, E_nuc-n_elec*μ, n_elec
end

# ── dot_operator, expand_basis!, sketched_rr, sketched_rayleigh_ritz are in TensorTrains ──

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

    palette = Makie.wong_colors()

    fig = Figure(size=(860, 430), fontsize=12)

    ylabel_left = isnothing(e_ref) ? "E  (Ha)" : "|E − Eref|  (Ha)"
    title_left  = isnothing(e_ref) ? "Ground state energy" : "Ground state energy error"
    yscale_left = isnothing(e_ref) ? identity : log10

    ax1 = Axis(fig[1, 1], xlabel="Iteration count",
               ylabel=ylabel_left, title=title_left, yscale=yscale_left)
    ax2 = Axis(fig[1, 2], xlabel="Iteration count",
               ylabel="Residual", title="Residual convergence", yscale=log10)

    offset = 0
    for (i, (label, history)) in enumerate(stages)
        iters       = [h.iter + offset     for h in history]
        vals_sketch = [h.e_sketch          for h in history]
        vals_true   = [h.e_true            for h in history]
        res_sketch  = [h.res_sketch        for h in history]
        res_sample  = [h.res_sample        for h in history]

        c = palette[i]

        if isnothing(e_ref)
            scatterlines!(ax1, iters, vals_sketch, color=c)
            scatterlines!(ax1, iters, vals_true,   color=c, linestyle=:dash)
        else
            nan_neg(v) = v > 0 ? v : NaN
            scatterlines!(ax1, iters, abs.(vals_sketch .- e_ref), color=c)
            scatterlines!(ax1, iters, nan_neg.(vals_true   .- e_ref), color=c, linestyle=:dash)
        end
        scatterlines!(ax2, iters, res_sketch, color=c)
        scatterlines!(ax2, iters, res_sample, color=c, linestyle=:dash)

        # DMRG reference line for this stage (left panel only, clipped to stage x-range)
        if !isnothing(dmrg_refs) && i <= length(dmrg_refs) && !isnothing(dmrg_refs[i])
            val = isnothing(e_ref) ? dmrg_refs[i] : dmrg_refs[i] - e_ref
            val > 0 && lines!(ax1, [iters[1], iters[end]], [val, val], color=c, linestyle=:dot, linewidth=1.5)
        end

        offset = iters[end]
    end

    # Single shared legend below both panels: one group for stages, one for line types
    stage_elems  = [LineElement(color=palette[i], linewidth=2) for i in eachindex(stages)]
    stage_labels = [label for (label, _) in stages]
    style_elems  = [LineElement(color=:black, linewidth=2, linestyle=:solid),
                    LineElement(color=:black, linewidth=2, linestyle=:dash)]
    style_labels = ["Sketched", "True / Sampled"]
    if !isnothing(dmrg_refs) && any(!isnothing, dmrg_refs)
        push!(style_elems,  LineElement(color=:black, linewidth=1.5, linestyle=:dot))
        push!(style_labels, "DMRG bound")
    end

    Legend(fig[2, 1:2],
           [stage_elems,  style_elems],
           [stage_labels, style_labels],
           ["Stage", "Line type"],
           orientation  = :horizontal,
           tellwidth    = false,
           framevisible = false)

    isnothing(filename) || save(filename, fig)
    return fig
end


# ── Hamiltonian assembly ──────────────────────────────────────────────────────

H, E_nuc, n_elec = load_or_build_mpo(FCIDUMP, n_cas, tol_mpo, 0.2)


# Reference Slater determinant (first n_elec spin-orbitals occupied)
ψ0 = slater(n_elec, 2n_cas)

# Particle number operator
N = part_num(ψ0.ttv_dims)

# Sanity check: Hartree-Fock energy
E_hf = dot(ψ0, H * ψ0) + E_nuc
println("HF energy (active space, n_cas=$n_cas): $E_hf")
# Reference DMRG ground state: ≈ -7.9836 Ha (full active space)

# ── Driver ────────────────────────────────────────────────────────────────────

seed      = rand(Int)
block_rks = 8
orthogonal = true

d1    = 10     # Krylov basis dimension
rmax1 = 10     # max TT rank for sketching and Ritz vectors

d2    = 10     # Krylov basis dimension
rmax2 = 20     # max TT rank

d3    = 10     # Krylov basis dimension
rmax3 = 20     # max TT rank

e1, ψ_rr, hist1 = sketched_rayleigh_ritz(H, ψ0, d1, rmax1;
                    orthogonal=orthogonal, block_rks=block_rks, seed=seed, k_trunc=10,
                    e_nuc=E_nuc)
println("Sketched ground state:  ", e1)
println("True Ritz:  ", dot_operator(ψ_rr,H,ψ_rr) + E_nuc,
        ";  Error: ", dot_operator(ψ_rr,H,ψ_rr) + E_nuc - e_tot)
println("Particle number:  ", dot_operator(ψ_rr,N,ψ_rr))

seed = seed + 1
e2, ψ_rr, hist2 = sketched_rayleigh_ritz(H, tt_rounding(ψ_rr, rmax=rmax2), d2, rmax2;
                    orthogonal=orthogonal, block_rks=block_rks, seed=seed, k_trunc=10,
                    e_nuc=E_nuc)
println("Sketched ground state:  ", e2)
println("True Ritz:  ", dot_operator(ψ_rr,H,ψ_rr) + E_nuc,
        ";  Error: ", dot_operator(ψ_rr,H,ψ_rr) + E_nuc - e_tot)
println("Particle number:  ", dot_operator(ψ_rr,N,ψ_rr))


seed = seed + 1
e3, ψ_rr, hist3 = sketched_rayleigh_ritz(H, tt_rounding(ψ_rr, rmax=rmax3), d3, rmax3;
                    orthogonal=orthogonal, block_rks=block_rks, seed=seed, k_trunc=10,
                    e_nuc=E_nuc)
println("Sketched ground state:  ", e3)
println("True Ritz:  ", dot_operator(ψ_rr,H,ψ_rr) + E_nuc,
        ";  Error: ", dot_operator(ψ_rr,H,ψ_rr) + E_nuc - e_tot)
println("Particle number:  ", dot_operator(ψ_rr,N,ψ_rr))

# d    = 10     # Krylov basis dimension
# rmax = 100     # max TT rank
# seed = seed + 1

# e, ψ_rr, hist4 = sketched_rayleigh_ritz(H, tt_rounding(ψ_rr, rmax=rmax), d, rmax;
#                     orthogonal=orthogonal, block_rks=block_rks, seed=seed, k_trunc=5,
#                     e_nuc=E_nuc)
# println("Sketched ground state:  ", e)
# println("True Ritz:  ", dot_operator(ψ_rr,H,ψ_rr) + E_nuc,
#         ";  Error: ", dot_operator(ψ_rr,H,ψ_rr) + E_nuc - e_tot)
# println("Particle number:  ", dot_operator(ψ_rr,N,ψ_rr))

v = tt_up_rks(ψ0, rmax1;ϵ_wn=1e-2)
E_tt1, ψ_tt1, _ = dmrg_eigsolv(H,v;schedule=dmrg_schedule_default(rmax=rmax1)) 
e_tt1 = E_tt1[end]

v = tt_up_rks(ψ_tt1, rmax2;ϵ_wn=1e-2)
E_tt2, ψ_tt2, _ = dmrg_eigsolv(H,v;schedule=dmrg_schedule_default(rmax=rmax2), verbose=false) 
e_tt2 = E_tt2[end]

v = tt_up_rks(ψ_tt2, rmax3;ϵ_wn=1e-2)
E_tt3, ψ_tt3, _ = dmrg_eigsolv(H,v;schedule=dmrg_schedule_default(rmax=rmax3), verbose=false) 
e_tt3 = E_tt3[end]

using CairoMakie

# ── Convergence plot (all stages combined) ────────────────────────────────────
plot_sRR_convergence([
        ("rmax=$(rmax1)",  hist1),
        ("rmax=$(rmax2)",  hist2),
        ("rmax=$(rmax3)",  hist3),
        # ("rmax=$(rmax4)", hist4),
    ]; e_ref=e_tot,
       # dmrg_refs=[e_tt + E_nuc],
       # dmrg_refs=[e_tt + E_nuc, e_tt3 + E_nuc],
       dmrg_refs=[e_tt1 + E_nuc, e_tt2 + E_nuc, e_tt3 + E_nuc],
       # dmrg_refs=[e_tt + E_nuc, e_tt2 + E_nuc, e_tt3 + E_nuc, e_tt4 + E_nuc],
       filename="convergence.pdf")


