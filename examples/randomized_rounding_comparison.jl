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
Randomized Rounding Comparison Experiments

Compares different rounding methods on perturbed `base_rank` tensor trains:
1. Deterministic rounding (tt_rounding)
2. Randomized rounding with orthogonal sketching (block_rks = 1, 2, 4)
3. Randomized rounding with non-orthogonal sketching (block_rks = 1, 2, 4)
4. STTA with orthogonal sketching (block_rks = 1, 2, 4)
5. STTA with non-orthogonal sketching (block_rks = 1, 2, 4)

Tests error vs perturbation strength for a base tensor of rank base_rank.
"""

"""
Experiment 1: Error vs Perturbation Strength for Base
"""
function synthetic_experiment(;
    N = 20,                                    # Tensor train dimension
    d = 4,                                     # Physical dimension per core
    base_rank = 1,                             # Rank of vector to be rounded
    perturbation_strengths = 10.0.^(-6:-1),    # ε values to test
    perturbation_rank = 10,                    # Perturbation rank
    block_rks_list = [1, 2, 4],                # Block ranks to test
    n_realizations = 10,                       # Realizations for randomized methods
    seed = 1234,
    force_rerun = false
)
    # Check if results already exist
    mkpath("out/randomized_rounding")
    filename = "out/randomized_rounding/rank($(base_rank)+$(perturbation_rank))_N$(N)_d$(d).json"

    if !force_rerun && isfile(filename)
        println("=== Perturbed Rank-$(base_rank) Experiment ===")
        println("Results already exist at: $filename")
        println("Loading existing results... (use force_rerun=true to recompute)")

        # Load and return existing results
        results = JSON3.read(read(filename, String), Dict{String,Any})

        # Verify parameters match
        params_match = (
            results["N"] == N &&
            results["d"] == d &&
            results["base_rank"] == base_rank &&
            results["perturbation_rank"] == perturbation_rank &&
            collect(results["perturbation_strengths"]) == collect(perturbation_strengths) &&
            collect(results["block_rks_list"]) == collect(block_rks_list) &&
            results["n_realizations"] == n_realizations
        )

        if !params_match
            println("WARNING: Loaded parameters do not match current parameters!")
            println("Loaded:  N=$(results["N"]), d=$(results["d"]), pert_rank=$(results["perturbation_rank"])")
            println("Current: N=$N, d=$d, pert_rank=$perturbation_rank")
            println("Loaded:  block_rks=$(collect(results["block_rks_list"])), n_real=$(results["n_realizations"])")
            println("Current: block_rks=$block_rks_list, n_real=$n_realizations")
            println("Rerunning experiment with current parameters...")
            # Fall through to run experiment
        else
            println("Parameters match! Using cached results.")

            # Convert arrays to proper format
            ε_list = collect(results["perturbation_strengths"])
            block_rks_loaded = collect(results["block_rks_list"])

            # Convert nested arrays to matrices/vectors
            for key in keys(results)
                if key ∉ ["N", "d", "base_rank", "perturbation_rank", "perturbation_strengths", "block_rks_list", "n_realizations", "seed"]
                    method_data = results[key]
                    if haskey(method_data, "errors")
                        arr = method_data["errors"]
                        if arr isa AbstractVector && length(arr) > 0 && arr[1] isa AbstractVector
                            # Convert to matrix (ε_index x realization)
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
            create_perturbation_plot(results, ε_list, block_rks_loaded)

            return results
        end
    end

    println("=== Perturbed Rank-$(base_rank) Experiment ===")
    println("Dimension: N = $N, d = $d")
    println("Base rank: $base_rank")
    println("Pertubation rank: $perturbation_rank")
    println("Perturbation strengths: $perturbation_strengths")
    println("Block ranks: $block_rks_list")
    println("Realizations: $n_realizations")
    println()

    dims = ntuple(i -> d, N)

    # Storage for results
    results = Dict{String,Any}()
    results["N"] = N
    results["d"] = d
    results["base_rank"] = base_rank
    results["perturbation_rank"] = perturbation_rank
    results["perturbation_strengths"] = perturbation_strengths
    results["block_rks_list"] = block_rks_list
    results["n_realizations"] = n_realizations
    results["seed"] = seed

    # Create base rank-1 tensor and noise once
    println("--- Creating base tensors ---")
    x0 = rand_tt(Float64, dims, base_rank; orthogonal=false, normalise=true, stable=false)
    x0 = x0/norm(x0)
    noise = rand_tt(Float64, dims, perturbation_rank; orthogonal=false, normalise=true, stable=false)
    noise = noise / norm(noise)
    x_perturbed = [ε*noise + x0 for ε in perturbation_strengths]
    for i in eachindex(perturbation_strengths)
        x_perturbed[i] = x_perturbed[i] / norm(x_perturbed[i])
    end
    println("Created x0 (rank-$base_rank) and perturbations (rank-$perturbation_rank)")
    println()

    # Initialize method storage
    methods = Dict{String,Any}()

    # Deterministic rounding (single realization)
    println("--- Deterministic Rounding ---")
    det_errors = Float64[]

    for (i_ε, ε) in enumerate(perturbation_strengths)
        print("  ε = $(@sprintf("%.1e", ε)): ")

        # Deterministic rounding
        x = x_perturbed[i_ε]
        x_rounded = tt_rounding(x; tol=1e-16, rmax=base_rank)
        error = norm(x_rounded - x)
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

            # Storage: [ε_index, realization]
            errors = zeros(length(perturbation_strengths), n_realizations)

            timer = TimerOutput()
            for (i_ε, ε) in enumerate(perturbation_strengths)
                print("    ε = $(@sprintf("%.1e", ε)): ")

                for real = 1:n_realizations
                    sketch_seed = seed + real + 1000*i_ε + 10000*block_rks

                    x = x_perturbed[i_ε]
                    x_rounded = ttrand_rounding([1, ε], [x0, noise], base_rank;
                                               orthogonal=orthogonal,
                                               block_rks=block_rks,
                                               seed=sketch_seed,
                                               timer=timer)
            # Display timing statistics

                    errors[i_ε, real] = norm(x_rounded - x)
                    print(".")
                end
                println()
            end
            # show(timer)
            # println("\n")


            median_errors = [median(errors[i, :]) for i in 1:length(perturbation_strengths)]
            q25_errors = [quantile(errors[i, :], 0.25) for i in 1:length(perturbation_strengths)]
            q75_errors = [quantile(errors[i, :], 0.75) for i in 1:length(perturbation_strengths)]
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

            # Storage: [ε_index, realization]
            errors = zeros(length(perturbation_strengths), n_realizations)
            timer = TimerOutput()

            for (i_ε, ε) in enumerate(perturbation_strengths)
                print("    ε = $(@sprintf("%.1e", ε)): ")

                for real = 1:n_realizations
                    sketch_seed_left = seed + real + 1000*i_ε + 10000*block_rks
                    sketch_seed_right = seed + real + 1000*i_ε + 10000*block_rks + 100000

                    x = x_perturbed[i_ε]
                    x_rounded = stta([1, ε], [x0, noise], base_rank;
                                    orthogonal=orthogonal,
                                    block_rks=block_rks,
                                    seed_left=sketch_seed_left,
                                    seed_right=sketch_seed_right,
                                    timer=timer)

                    errors[i_ε, real] = norm(x_rounded - x)
                    print(".")
                end
                println()
            end

            # Display timing statistics
            # show(timer)
            # println("\n")

            median_errors = [median(errors[i, :]) for i in 1:length(perturbation_strengths)]
            q25_errors = [quantile(errors[i, :], 0.25) for i in 1:length(perturbation_strengths)]
            q75_errors = [quantile(errors[i, :], 0.75) for i in 1:length(perturbation_strengths)]
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
    create_perturbation_plot(results, perturbation_strengths, block_rks_list)
    
    return results
end

"""
Create plot showing error vs perturbation strength for all methods
"""
function create_perturbation_plot(results, ε_list, block_rks_list)
    mkpath("out/randomized_rounding/plots")

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

    p = plot(xlabel="Perturbation Strength ε",
             ylabel="Relative Error ||x̃ - x₀||/||x₀||",
             xscale=:log10, yscale=:log10,
             legend=:bottomright,
             ylims=(minimum(results["perturbation_strengths"]), 1),
             size=(800, 600),
             title="Randomized Rounding: Error vs Perturbation (N=$(results["N"]), d=$(results["d"]))",
             dpi=150)

    # Plot deterministic rounding (reference line)
    det_data = results["deterministic"]
    plot!(p, ε_list, det_data["median_error"],
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
            plot!(p, ε_list, med,
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
            plot!(p, ε_list, med,
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
            plot!(p, ε_list, med,
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
            plot!(p, ε_list, med,
                  yerror=(err_low, err_high),
                  label=(block_rks==1 ? "STTA Khatri-Rao (Gaussian)" : "STTA non-orth R=$block_rks"),
                  color=color,
                  linewidth=2,
                  linestyle=style_stta_nonorth,
                  marker=:utriangle,
                  markersize=4)
        end
    end

    # Add reference lines
    plot!(p, ε_list, ε_list, label="ε (reference)", color=:gray, linestyle=:dashdot, linewidth=1)

    display(p)
    sleep(5)
    savefig(p, "out/randomized_rounding/plots/rank($(results["base_rank"])+$(results["perturbation_rank"]))_N$(results["N"])_d$(results["d"]).png")
    println("Plot saved")
end

"""
Run the experiment
"""
function run_randomized_rounding_experiments(; force_rerun = false)
    println("="^70)
    println("RANDOMIZED ROUNDING COMPARISON EXPERIMENTS")
    println("="^70)

    results = synthetic_experiment(
        N = 50,
        d = 4,
        base_rank = 1,
        perturbation_strengths = 10.0.^(-6:-1),
        perturbation_rank = 50,
        block_rks_list = [1, 2, 4],
        n_realizations = 10,
        force_rerun = force_rerun
    )

    println("\n" * "="^70)
    println("EXPERIMENTS COMPLETED")
    println("="^70)

    return results
end

# Run experiments automatically when file is included or executed
println("Running randomized rounding comparison experiments...")
global randomized_rounding_results = run_randomized_rounding_experiments()
