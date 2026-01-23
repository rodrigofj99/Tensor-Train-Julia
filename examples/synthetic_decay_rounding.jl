using Revise
using TensorTrains
using LinearAlgebra
using Statistics
using JSON3
using Random
using Plots
using Printf
using TimerOutputs

"""
TT-Rounding Experiments

Compares different rounding methods on synthetic wavefunctions with increasing
target ranks. After randomized rounding, applies deterministic rounding to ensure fair
comparison at exact target ranks.

Methods tested:
1. Deterministic rounding (tt_rounding) - baseline
2. Randomized rounding (ttrand_rounding) + deterministic cleanup
3. STTA + deterministic cleanup

For orthogonal and non-orthogonal sketches with different block ranks.
"""

"""
Synthetic Rounding Experiment
"""
function synthetic_experiment(;
    N = 20,                                    # Number of orbitals
    decay_rate = .2,                           # Decay rate for multiplier applied to rand-TT singular values
    target_ranks = [5, 10, 15, 20, 25, 30],    # Target ranks to test
    block_rks_list = [1, 2, 4],                # Block ranks to test
    n_realizations = 10,                       # Realizations for randomized methods
    seed = 1234,
    force_rerun = false
)
    # Check if results already exist
    mkpath("out/synthetic")
    filename = "out/synthetic/d$(decay_rate)_N$(N).json"

    if !force_rerun && isfile(filename)
        println("=== Synthetic Experiment ===")
        println("Results already exist at: $filename")
        println("Loading existing results... (use force_rerun=true to recompute)")

        # Load and return existing results
        results = JSON3.read(read(filename, String), Dict{String,Any})

        # Verify parameters match
        params_match = (
            results["N"] == N &&
            results["decay_rate"] == decay_rate &&
            collect(results["target_ranks"]) == collect(target_ranks) &&
            collect(results["block_rks_list"]) == collect(block_rks_list) &&
            results["n_realizations"] == n_realizations
        )

        if !params_match
            println("WARNING: Loaded parameters do not match current parameters!")
            println("Loaded:  N=$(results["N"])")
            println("Current: N=$N, decay_rate=$decay_rate")
            println("Loaded:  target_ranks=$(collect(results["target_ranks"])), block_rks=$(collect(results["block_rks_list"]))")
            println("Current: target_ranks=$target_ranks, block_rks=$block_rks_list")
            println("Rerunning experiment with current parameters...")
            # Fall through to run experiment
        else
            println("Parameters match! Using cached results.")

            # Convert arrays to proper format
            ranks_loaded = collect(results["target_ranks"])
            block_rks_loaded = collect(results["block_rks_list"])

            # Convert nested arrays to matrices/vectors
            for key in keys(results)
                if key ∉ ["N", "decay_rate", "target_ranks", "block_rks_list", "n_realizations", "seed"]
                    method_data = results[key]
                    if haskey(method_data, "errors")
                        arr = method_data["errors"]
                        if arr isa AbstractVector && length(arr) > 0 && arr[1] isa AbstractVector
                            # Convert to matrix (rank_index x realization)
                            method_data["errors"] = hcat([collect(row) for row in arr]...)'
                        end
                    end
                    for stat_key in ["median_error", "q25_error", "q75_error"]
                        if haskey(method_data, stat_key)
                            method_data[stat_key] = collect(method_data[stat_key])
                        end
                    end
                end
            end

            # Recreate plots
            println("Recreating plots from existing data...")
            create_synthetic_plot(results, ranks_loaded, block_rks_loaded)

            return results
        end
    end

    println("=== Synthetic Rounding Experiment ===")
    println("Number of orbitals: N = $N")
    println("Singular values decay rate: decay_rate = $decay_rate")
    println("Target ranks: $target_ranks")
    println("Block ranks: $block_rks_list")
    println("Realizations: $n_realizations")
    println()

    # Storage for results
    results = Dict{String,Any}()
    results["N"] = N
    results["decay_rate"] = decay_rate
    results["target_ranks"] = target_ranks
    results["block_rks_list"] = block_rks_list
    results["n_realizations"] = n_realizations
    results["seed"] = seed

    println("--- Creating randomized vector ---")
    Random.seed!(seed)
    psi = rand_tt(ntuple(i->2, N), 200, orthogonal=true);
    v = tt_to_vidal(psi);
    tol = 1e-10;
    cutoff = ceil(Int, 1/decay_rate * log(1/tol))
    for i=1:length(v.Σ)
        if length(v.Σ[i]) <= cutoff
            v.Σ[i] .*= exp.( -decay_rate .* axes(v.Σ[i],1))
            v.cores[i+1] = v.cores[i+1]
        @show v.Σ[i]
    end
    psi = vidal_to_left_canonical(v)
    
    psi = psi / norm(psi)
    psi_norm = norm(psi)

    println("Created randomized state with $(length(psi.ttv_vec)) cores")
    println("Initial ranks: $(psi.ttv_rks)")
    println("Norm: $psi_norm")
    println()

    # Initialize method storage
    methods = Dict{String,Any}()

    # Deterministic rounding (baseline)
    println("--- Deterministic Rounding ---")
    det_errors = Float64[]

    for (i_r, target_rank) in enumerate(target_ranks)
        print("  target_rank = $target_rank: ")

        psi_rounded = tt_rounding(psi; tol=0.0, rmax=target_rank)
        error = norm(psi_rounded - psi) / psi_norm
        push!(det_errors, error)
        println("error = $(@sprintf("%.3e", error))")
    end

    methods["deterministic"] = Dict("errors" => det_errors, "median_error" => det_errors)

    # Randomized rounding with different configurations
    for orthogonal in [true, false]
        orth_str = orthogonal ? "orthogonal" : "non_orthogonal"
        println("\n--- Randomized Rounding ($orth_str) ---")

        for block_rks in block_rks_list
            method_name = "ttrand_$(orth_str)_blk$(block_rks)"
            println("  Block ranks = $block_rks")

            # Storage: [rank_index, realization]
            errors = zeros(length(target_ranks), n_realizations)

            timer = TimerOutput()
            for (i_r, target_rank) in enumerate(target_ranks)
                print("    target_rank = $target_rank: ")

                for real = 1:n_realizations
                    sketch_seed = seed + real + 1000*i_r + 10000*block_rks

                    # Randomized rounding
                    psi_rand = ttrand_rounding(psi, target_rank;
                                               orthogonal=orthogonal,
                                               block_rks=block_rks,
                                               seed=sketch_seed,
                                               timer=timer)

                    # Deterministic cleanup to exact target rank
                    psi_rounded = tt_rounding(psi_rand; tol=0.0, rmax=target_rank)

                    errors[i_r, real] = norm(psi_rounded - psi) / psi_norm
                    print(".")
                end
                println()
            end

            median_errors = [median(errors[i, :]) for i in 1:length(target_ranks)]
            q25_errors = [quantile(errors[i, :], 0.25) for i in 1:length(target_ranks)]
            q75_errors = [quantile(errors[i, :], 0.75) for i in 1:length(target_ranks)]
            methods[method_name] = Dict(
                "errors" => errors,
                "median_error" => median_errors,
                "q25_error" => q25_errors,
                "q75_error" => q75_errors
            )
        end
    end

    # STTA with different configurations
    for orthogonal in [true, false]
        orth_str = orthogonal ? "orthogonal" : "non_orthogonal"
        println("\n--- STTA ($orth_str) ---")

        for block_rks in block_rks_list
            method_name = "stta_$(orth_str)_blk$(block_rks)"
            println("  Block ranks = $block_rks")

            # Storage: [rank_index, realization]
            errors = zeros(length(target_ranks), n_realizations)

            timer = TimerOutput()
            for (i_r, target_rank) in enumerate(target_ranks)
                print("    target_rank = $target_rank: ")

                for real = 1:n_realizations
                    sketch_seed_left = seed + real + 1000*i_r + 10000*block_rks
                    sketch_seed_right = seed + real + 1000*i_r + 10000*block_rks + 100000

                    # STTA
                    psi_stta = stta(psi, target_rank;
                                   orthogonal=orthogonal,
                                   block_rks=block_rks,
                                   seed_left=sketch_seed_left,
                                   seed_right=sketch_seed_right,
                                   timer=timer)

                    # Deterministic cleanup to exact target rank
                    psi_rounded = tt_rounding(psi_stta; tol=0.0, rmax=target_rank)

                    errors[i_r, real] = norm(psi_rounded - psi) / psi_norm
                    print(".")
                end
                println()
            end

            median_errors = [median(errors[i, :]) for i in 1:length(target_ranks)]
            q25_errors = [quantile(errors[i, :], 0.25) for i in 1:length(target_ranks)]
            q75_errors = [quantile(errors[i, :], 0.75) for i in 1:length(target_ranks)]
            methods[method_name] = Dict(
                "errors" => errors,
                "median_error" => median_errors,
                "q25_error" => q25_errors,
                "q75_error" => q75_errors
            )
        end
    end

    # Store all methods
    for (method_name, method_data) in methods
        results[method_name] = method_data
    end

    # Save results
    open(io -> JSON3.write(io, results, allow_inf=true), filename, "w")
    println("Results saved to: $filename")

    # Create plots
    println("\n--- Creating plots ---")
    create_synthetic_plot(results, target_ranks, block_rks_list)

    return results
end

"""
Create plot showing error vs target rank for all methods
"""
function create_synthetic_plot(results, rank_list, block_rks_list)
    mkpath("out/synthetic/plots")

    # Color scheme for block ranks - dynamically assign colors
    color_palette = [:blue, :red, :green, :orange, :purple, :brown, :pink, :gray]
    colors_blk = Dict(block_rks_list[i] => color_palette[mod1(i, length(color_palette))]
                      for i in 1:length(block_rks_list))

    # Line styles
    style_det = :solid
    style_ttrand_orth = :solid
    style_ttrand_nonorth = :dash
    style_stta_orth = :dot
    style_stta_nonorth = :dashdot

    p = plot(xlabel="Target Rank",
             ylabel="Relative Error ||ψ̃ - ψ||/||ψ||",
             yscale=:log10,
             legend=:topright,
             size=(800, 600),
             title="Synthetic Rounding (N=$(results["N"]), n_elec=$(results["n_elec"]))",
             dpi=150)

    # Plot deterministic rounding (reference line)
    det_data = results["deterministic"]
    plot!(p, rank_list, det_data["median_error"],
          label="Deterministic",
          color=:black,
          linewidth=3,
          linestyle=style_det)

    # Plot randomized methods
    for block_rks in block_rks_list
        color = colors_blk[block_rks]

        # ttrand orthogonal
        method_name = "ttrand_orthogonal_blk$(block_rks)"
        if haskey(results, method_name)
            data = results[method_name]
            med = data["median_error"]
            err_low = med .- data["q25_error"]
            err_high = data["q75_error"] .- med
            plot!(p, rank_list, med,
                  yerror=(err_low, err_high),
                  label=(block_rks==1 ? "ttrand Khatri-Rao (spherical)" : "ttrand orth R=$block_rks"),
                  color=color,
                  linewidth=2,
                  linestyle=style_ttrand_orth,
                  marker=:circle,
                  markersize=4)
        end

        # ttrand non-orthogonal
        method_name = "ttrand_non_orthogonal_blk$(block_rks)"
        if haskey(results, method_name)
            data = results[method_name]
            med = data["median_error"]
            err_low = med .- data["q25_error"]
            err_high = data["q75_error"] .- med
            plot!(p, rank_list, med,
                  yerror=(err_low, err_high),
                  label=(block_rks==1 ? "ttrand Khatri-Rao (Gaussian)" : "ttrand non-orth R=$block_rks"),
                  color=color,
                  linewidth=2,
                  linestyle=style_ttrand_nonorth,
                  marker=:square,
                  markersize=4)
        end

        # STTA orthogonal
        method_name = "stta_orthogonal_blk$(block_rks)"
        if haskey(results, method_name)
            data = results[method_name]
            med = data["median_error"]
            err_low = med .- data["q25_error"]
            err_high = data["q75_error"] .- med
            plot!(p, rank_list, med,
                  yerror=(err_low, err_high),
                  label=(block_rks==1 ? "STTA Khatri-Rao (spherical)" : "STTA orth R=$block_rks"),
                  color=color,
                  linewidth=2,
                  linestyle=style_stta_orth,
                  marker=:diamond,
                  markersize=4)
        end

        # STTA non-orthogonal
        method_name = "stta_non_orthogonal_blk$(block_rks)"
        if haskey(results, method_name)
            data = results[method_name]
            med = data["median_error"]
            err_low = med .- data["q25_error"]
            err_high = data["q75_error"] .- med
            plot!(p, rank_list, med,
                  yerror=(err_low, err_high),
                  label=(block_rks==1 ? "STTA Khatri-Rao (Gaussian)" : "STTA non-orth R=$block_rks"),
                  color=color,
                  linewidth=2,
                  linestyle=style_stta_nonorth,
                  marker=:utriangle,
                  markersize=4)
        end
    end

    display(p)
    sleep(5)
    savefig(p, "out/synthetic/plots/N$(results["N"])_nelec$(results["n_elec"]).png")
    println("Plot saved")
end

"""
Run the experiment
"""
function run_synthetic_experiments(; force_rerun = false)
    println("="^70)
    println("SYNTHETIC ROUNDING EXPERIMENTS")
    println("="^70)

    results = synthetic_experiment(
        N = 50,
        decay_rate = .2,
        target_ranks = [5, 10, 15, 20, 25, 30],
        block_rks_list = [1, 2, 4, 8],
        n_realizations = 5,
        force_rerun = force_rerun
    )

    println("\n" * "="^70)
    println("EXPERIMENTS COMPLETED")
    println("="^70)

    return results
end

# Run experiments automatically when file is included or executed
println("Running synthetic rounding experiments...")
global synthetic_results = run_synthetic_experiments()
