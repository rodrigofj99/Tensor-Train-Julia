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


"""
    save_sRR_results(filename, stages, seeds, schedule, λ)

Serialize sRR output to a JSON file.  The file stores:
- `final_energy`: the last Ritz value (including `e_nuc`)
- `seeds`: actual seeds used per stage
- `schedule`: the stage parameters (NamedTuple fields)
- `stages`: one object per stage with `label` and `history` array
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
                lines!(ax1, iters, vals_true,   color=c)
                lines!(ax1, iters, vals_sketch, color=c, linestyle=:dash)
            else
                nan_pos(v) = v > 0 ? v : NaN
                lines!(ax1, iters, nan_pos.(vals_true   .- e_ref), color=c)
                lines!(ax1, iters, abs.(vals_sketch .- e_ref),     color=c, linestyle=:dash)
            end
            lines!(ax2, iters, res_sample.^2, color=c)
            lines!(ax2, iters, res_sketch.^2, color=c, linestyle=:dash)

            # DMRG reference line for this stage (left panel only)
            if !isnothing(dmrg_refs) && i <= length(dmrg_refs) && !isnothing(dmrg_refs[i])
                val = isnothing(e_ref) ? dmrg_refs[i] : dmrg_refs[i] - e_ref
                val > 0 && lines!(ax1, [iters[1], iters[end]], [val, val],
                                  color=c, linestyle=:dot, linewidth=1.5)
            end

            offset = iters[end]
        end

        # Line-type legend: solid = estimated/true, dashed = sketched
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

block_rks  = 16
orthogonal = true

schedule = [
    (d=10, rmax=10,  sketch_size=200,  k_trunc=5, label="rmax=10"),
    (d=10, rmax=15,  sketch_size=200,  k_trunc=5, label="rmax=15"),
    (d=10, rmax=20,  sketch_size=200,  k_trunc=5, label="rmax=20"),
    (d=10, rmax=30,  sketch_size=200,  k_trunc=5, label="rmax=30"),
    (d=10, rmax=40,  sketch_size=200,  k_trunc=5, label="rmax=40"),
    (d=10, rmax=50,  sketch_size=200,  k_trunc=5, label="rmax=50"),
    (d=10, rmax=75,  sketch_size=200,  k_trunc=5, label="rmax=75"),
    (d=10, rmax=100, sketch_size=200,  k_trunc=5, label="rmax=100")
]

e, ψ_rr, stages, seeds = sketched_rayleigh_ritz(H, ψ0, schedule;
                    orthogonal=orthogonal, block_rks=block_rks,
                    seed=rand(Int), e_nuc=E_nuc,
                    validate_seed=true)

println("Final energy:    ", e)
println("True Ritz:       ", dot_operator(ψ_rr, H, ψ_rr) + E_nuc,
        ";  Error: ", dot_operator(ψ_rr, H, ψ_rr) + E_nuc - e_tot)
println("Particle number: ", dot_operator(ψ_rr, N, ψ_rr))

save_sRR_results("lih_sRR_ncas$(n_cas).json", stages, seeds, schedule, e)

# ── Optional DMRG reference ───────────────────────────────────────────────────
rmax_vals = [s.rmax for s in schedule]
v = tt_up_rks(ψ0, rmax_vals[1]; ϵ_wn=1e-2)
E_tt, ψ_tt, _ = dmrg_eigsolv(H, v; schedule=dmrg_schedule_default(rmax=rmax_vals[1]))
dmrg_refs = [E_tt[end] + E_nuc]
for r in rmax_vals[2:end]
    global v = tt_up_rks(ψ_tt, r; ϵ_wn=1e-2)
    global E_tt, ψ_tt, _ = dmrg_eigsolv(H, v; schedule=dmrg_schedule_default(rmax=r), verbose=false)
    push!(dmrg_refs, E_tt[end] + E_nuc)
end

using CairoMakie

# ── Convergence plot ──────────────────────────────────────────────────────────
plot_sRR_convergence(stages;
    e_ref=e_tot,
    dmrg_refs=nothing, #dmrg_refs,
    filename="convergence.pdf")


