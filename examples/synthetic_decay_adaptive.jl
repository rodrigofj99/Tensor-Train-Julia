using TensorTrains
using LinearAlgebra
using Random
using Printf
using Statistics

"""
Same synthetic decaying-spectrum TT generator as `examples/synthetic_decay_rounding.jl`:
build a `rand_tt` of binary modes at rank 200, convert to Vidal form, multiply each bond's
singular values by `exp(-decay_rate · index)` to impose the decay, fold back, normalise.
"""
function make_synthetic(N::Int, decay_rate::Float64, seed::Int)
    Random.seed!(seed)
    psi = rand_tt(ntuple(i -> 2, N), 200, orthogonal=true)
    v = tt_to_vidal(psi)
    tol = 1e-10
    cutoff = ceil(Int, 1/decay_rate * log(1/tol))
    for i = 1:length(v.Σ)
        if length(v.Σ[i]) <= cutoff
            v.Σ[i] .*= exp.(-decay_rate .* axes(v.Σ[i], 1))
        end
    end
    psi = vidal_to_left_canonical(v)
    psi = psi / norm(psi)
    return psi
end

"""
For each (N, decay) configuration sweep ε and report median (across realisations) of
the achieved max rank, the achieved relative error, and the deterministic error at the
same rank for context.
"""
function adaptive_vs_deterministic(; N::Int = 20, decay_rate::Float64 = 0.5,
                                     εs = (1e-2, 1e-4, 1e-6, 1e-8),
                                     n_realisations::Int = 3,
                                     seed::Int = 1234)
    psi = make_synthetic(N, decay_rate, seed)
    psi_norm = norm(psi)

    println("="^72)
    println("Synthetic state — N=$N, decay=$decay_rate, max input rank=$(maximum(psi.ttv_rks))")
    println("="^72)
    @printf "%-9s | %-26s | %-26s | %-22s\n" "ε" "adaptive (max rk, err)" "det at adaptive's rk" "theory rank ~log(1/ε)/decay"
    println("-"^99)

    for ε in εs
        errs = Float64[]
        ranks = Int[]
        for r = 1:n_realisations
            ŷ = ttrand_rounding_adaptive(psi, ε; seed = seed + 1000*r)
            push!(errs, norm(psi - ŷ) / psi_norm)
            push!(ranks, maximum(ŷ.ttv_rks))
        end
        med_err = median(errs)
        med_rk = round(Int, median(ranks))

        psi_det = tt_rounding(psi; tol=0.0, rmax=med_rk)
        det_err = norm(psi - psi_det) / psi_norm

        rk_theory = ceil(Int, log(1/ε) / decay_rate)

        @printf "%-9.0e | rk %3d  err %.2e   | rk %3d  err %.2e   | rk %3d\n" ε med_rk med_err med_rk det_err rk_theory
    end
    println()
end

# === run a few configurations ===
adaptive_vs_deterministic(N = 12, decay_rate = 0.5)
adaptive_vs_deterministic(N = 20, decay_rate = 0.5)
adaptive_vs_deterministic(N = 20, decay_rate = 0.2)
