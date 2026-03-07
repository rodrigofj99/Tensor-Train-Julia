using Revise
using TensorTrains
using LinearAlgebra
using Statistics
using JSON3
using Random
using CairoMakie
using LaTeXStrings
using Printf
using TimerOutputs

"""
Randomized Rounding Comparison Experiments

Compares different rounding methods on perturbed `base_rank` tensor trains:
1. Deterministic rounding (tt_rounding)
2. Randomized rounding with orthogonal sketching
3. Randomized rounding with non-orthogonal sketching 
4. STTA with orthogonal sketching 
5. STTA with non-orthogonal sketching

Tests error vs Noise Level for a base tensor of rank base_rank.
"""

"""
Experiment 1: Error vs Noise Level for Base
"""
function synthetic_experiment(;
    N = 20,                                    # Tensor train dimension
    d = 4,                                     # Physical dimension per core
    base_rank = 8,                             # Target rank for rounding
    n_summands = 8,                            # Number of rank-1 summands to create base tensor
    perturbation_strengths = 10.0.^(-6:-1),    # ε values to test
    perturbation_rank = 10,                    # Perturbation rank
    block_rks_list = [1, 4, 8],                # Block ranks to test
    n_realizations = 10,                       # Realizations for randomized methods
    seed = 1234,
    force_rerun = false,
    dir = "out/randomized_rounding"
)
    # Check if results already exist
    mkpath(dir)
    filename = "$dir/rank$(base_rank)_summands$(n_summands)_N$(N)_d$(d).json"

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
            results["n_summands"] == n_summands &&
            results["perturbation_rank"] == perturbation_rank &&
            collect(results["perturbation_strengths"]) == collect(perturbation_strengths) &&
            collect(results["block_rks_list"]) == collect(block_rks_list) &&
            results["n_realizations"] == n_realizations
        )

        if !params_match
            println("WARNING: Loaded parameters do not match current parameters!")
            println("Loaded:  N=$(results["N"]), d=$(results["d"]), base_rank=$(results["base_rank"]), n_summands=$(results["n_summands"])")
            println("Current: N=$N, d=$d, base_rank=$base_rank, n_summands=$n_summands")
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
                if key ∉ ["N", "d", "base_rank", "n_summands", "perturbation_rank", "perturbation_strengths", "block_rks_list", "n_realizations", "seed"]
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
            create_perturbation_plot(results, dir=dir)

            return results
        end
    end

    println("=== Perturbed Rank-$(base_rank) Experiment ===")
    println("Dimension: N = $N, d = $d")
    println("Base rank: $base_rank (from $n_summands summands)")
    println("Pertubation rank: $perturbation_rank")
    println("Noise Levels: $perturbation_strengths")
    println("Block ranks: $block_rks_list")
    println("Realizations: $n_realizations")
    println()

    dims = ntuple(i -> d, N)

    # Storage for results
    results = Dict{String,Any}()
    results["N"] = N
    results["d"] = d
    results["base_rank"] = base_rank
    results["n_summands"] = n_summands
    results["perturbation_rank"] = perturbation_rank
    results["perturbation_strengths"] = perturbation_strengths
    results["block_rks_list"] = block_rks_list
    results["n_realizations"] = n_realizations
    results["seed"] = seed

    # Create base tensor as sum of n_summands tensors, each of rank base_rank÷n_summands
    println("--- Creating base tensors ---")
    Random.seed!(seed)
    summand_rank = base_rank ÷ n_summands
    if summand_rank * n_summands != base_rank
        error("base_rank ($base_rank) must be divisible by n_summands ($n_summands)")
    end
    
    x0_summands = [rand_tt(Float64, dims, summand_rank; orthogonal=false, normalise=true, stable=false) for _ in 1:n_summands]
    x0 = sum(x0_summands)
    x0 = x0/norm(x0)
    println("Created x0 as sum of $n_summands rank-$summand_rank tensors (target rank: $base_rank)")

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
    for orthogonal in [true]#, false]
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

    # STTA with different configurations
    for orthogonal in [true]#, false]
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
    create_perturbation_plot(results)
    
    return results
end

"""
Create plot showing error vs Noise Level for all methods
"""
function create_perturbation_plot(results; dir = "out/randomized_rounding")
    # Ensure proper types for plotting
    ε_list = Float64.(results["perturbation_strengths"])
    block_rks_list = Int.(results["block_rks_list"])
    mkpath("$dir/plots")

    # Configure CairoMakie for publication-quality plots
    CairoMakie.activate!(type = "pdf")
    
    # Define consistent styling theme
    PLOT_THEME = Theme(
        fontsize = 12,
        font = "Computer Modern",
        linewidth = 2,
        markersize = 8,
        Axis = (
            titlesize = 12,
            xlabelsize = 11,
            ylabelsize = 11,
            xticklabelsize = 10,
            yticklabelsize = 10,
            spinewidth = 1,
            xtickwidth = 1,
            ytickwidth = 1,
            xgridwidth = 0.5,
            ygridwidth = 0.5,
            xgridcolor = (:gray, 0.3),
            ygridcolor = (:gray, 0.3)
        ),
        Legend = (
            framevisible = true,
            backgroundcolor = (:white, 0.9),
            framecolor = :gray,
            framewidth = 1,
            padding = (8, 8, 6, 6),
            rowgap = 8,
            colgap = 12,
            labelsize = 10
        )
    )
    
    with_theme(PLOT_THEME) do
        # Base palettes for dynamic generation
        base_colors = [:blue, :orange, :green, :red, :purple, :cyan, :brown, :magenta]
        base_markers = [:circle, :rect, :diamond, :utriangle, :dtriangle, :star5, :xcross, :hexagon]

        # Dynamically build mapping dictionaries and legend elements
        colors_blk = Dict{Int, Symbol}()
        markers_blk = Dict{Int, Symbol}()
        color_elements = MarkerElement[]
        color_labels = LaTeXString[]

        for (i, rk) in enumerate(block_rks_list)
            c = base_colors[mod1(i, length(base_colors))]
            m = base_markers[mod1(i, length(base_markers))]
            colors_blk[rk] = c
            markers_blk[rk] = m
            
            push!(color_elements, MarkerElement(marker = m, color = c, markersize = 10, strokecolor = :transparent))
            label_str = rk == 1 ? "Rank 1 (Khatri-Rao)" : "Rank $rk"
            push!(color_labels, LaTeXString(label_str))
        end

        # Create title with interpolated values
        N_val = results["N"]
        d_val = results["d"]
        title_str = "Randomized Rounding: Error vs Perturbation (N=$N_val, d=$(d_val))"
        
        fig = Figure(size = (800, 600))
        ax = Axis(fig[1, 1],
                  xlabel = L"Noise Level $\varepsilon$",
                  ylabel = LaTeXString("Relative Error"),
                  xscale = log10,
                  yscale = log10,
                  limits = (nothing, (1e-5, 1e3)),
                  xticks = ([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6], [L"10^{-1}", L"10^{-2}", L"10^{-3}", L"10^{-4}", L"10^{-5}", L"10^{-6}"]),
                  title = title_str)

        # Skip deterministic rounding - only plot randomized methods

        # Plot selected methods: RandOrth (ttrand orthogonal) and STTA orthogonal
        for block_rks in block_rks_list
            color = colors_blk[block_rks]
            marker = markers_blk[block_rks]

            for (linestyle, method_name) in [(:solid, "ttrand_orthogonal_blk"), (:dash, "stta_orthogonal_blk")]
                full_key = method_name*"$(block_rks)"
                if !haskey(results, full_key)
                    continue
                end

                data = results[full_key]
                med = Float64.(data["median_error"])
                err_low = Float64.(data["q25_error"])
                err_high = Float64.(data["q75_error"])

                color = colors_blk[block_rks]
                marker = markers_blk[block_rks]
                
                band!(ax, ε_list, err_low, err_high,
                        color=(color, 0.3))
                scatterlines!(ax, ε_list, med,
                        color=color,
                        linewidth=2,
                        linestyle=linestyle,
                        marker=marker,
                        markersize=5)
            end
        end

        # Add reference lines
        lines!(ax, ε_list, ε_list, label=L"$\varepsilon$ (reference)", color=:gray, linestyle=:dashdot, linewidth=1)
        
        # Create custom legend elements
        line_elements = [
            LineElement(color = :black, linestyle = nothing, linewidth = 2),
            LineElement(color = :black, linestyle = :dot, linewidth = 2)
        ]
        line_labels = [LaTeXString("RandOrth (solid)"), LaTeXString("STTA (dotted)")]
        
        ref_elements = [LineElement(color = :gray, linestyle = :dashdot, linewidth = 1)]
        ref_labels = [L"$\varepsilon$ (reference)"]
        
        # Combine all elements
        all_elements = [line_elements..., color_elements..., ref_elements...]
        all_labels = [line_labels..., color_labels..., ref_labels...]
        
        # Add horizontal legend below the plot
        Legend(fig[2, 1], all_elements, all_labels,
               orientation = :horizontal, 
               tellheight = true, 
               framevisible = true,
               halign = :center,
               nbanks = 1,  # Single row
               padding = (4, 4, 2, 2),
               colgap = 8,
               labelsize = 9)

        display(fig)
        
        # Save the plot
        save("$dir/plots/rank$(results["base_rank"])_summands$(results["n_summands"])_N$(results["N"])_d$(results["d"]).pdf", fig)
        println("Plot saved as PDF")
    end
end

"""
Create combined plot comparing both tensor structures
"""
function create_combined_plot(results1, results2; dir = "out/randomized_rounding")
    mkpath("$dir/plots")

    # Configure CairoMakie for publication-quality plots
    CairoMakie.activate!(type = "pdf")
    
    # Define consistent styling theme
    PLOT_THEME = Theme(
        fontsize = 11,
        font = "Computer Modern",
        linewidth = 2,
        markersize = 6,
        Axis = (
            titlesize = 11,
            xlabelsize = 10,
            ylabelsize = 10,
            xticklabelsize = 9,
            yticklabelsize = 9,
            spinewidth = 1,
            xtickwidth = 1,
            ytickwidth = 1,
            xgridwidth = 0.5,
            ygridwidth = 0.5,
            xgridcolor = (:gray, 0.3),
            ygridcolor = (:gray, 0.3)
        ),
        Legend = (
            framevisible = true,
            backgroundcolor = (:white, 0.9),
            framecolor = :gray,
            framewidth = 1,
            padding = (6, 6, 3, 3),
            rowgap = 0,
            colgap = 8,
            labelsize = 9
        )
    )
    
    with_theme(PLOT_THEME) do
        # Ensure proper types for plotting
        ε_list = Float64.(results1["perturbation_strengths"])
        block_rks_list = Int.(results1["block_rks_list"])
        
        # Base palettes for dynamic generation
        base_colors = [:blue, :orange, :green, :red, :purple, :cyan, :brown, :magenta]
        base_markers = [:circle, :rect, :diamond, :utriangle, :dtriangle, :star5, :xcross, :hexagon]
        
        # Dynamically build mapping dictionaries and legend elements
        colors_blk = Dict{Int, Symbol}()
        markers_blk = Dict{Int, Symbol}()
        color_elements = MarkerElement[]
        color_labels = LaTeXString[]

        for (i, rk) in enumerate(block_rks_list)
            c = base_colors[mod1(i, length(base_colors))]
            m = base_markers[mod1(i, length(base_markers))]
            colors_blk[rk] = c
            markers_blk[rk] = m
            
            push!(color_elements, MarkerElement(marker = m, color = c, markersize = 10, strokecolor = :transparent))
            label_str = rk == 1 ? "R=1 (Khatri-Rao)" : "R=$rk"
            push!(color_labels, LaTeXString(label_str))
        end

        fig = Figure(size = (624, 300))  # 6.5" wide, 4.5" tall (72 DPI equivalent)
        s = 16
        
        # Create two subplots side by side
        ax1 = Axis(fig[1, 1],
                   xlabel = L"Noise Level $\varepsilon$",
                   ylabel = L"Relative Error $\Vert \mathrm{trunc}_{16}(\mathbf{x}_\varepsilon) - \mathbf{x}_\varepsilon \Vert /\Vert \mathbf{x} \Vert $",
                   xscale = log10,
                   yscale = log10,
                   limits = (nothing, (1e-5, 1e3)),
                   xticks = ([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6], [L"10^{-1}", L"10^{-2}", L"10^{-3}", L"10^{-4}", L"10^{-5}", L"10^{-6}"]),
                   title = "Sum of $s Kronecker Bases")

        ax2 = Axis(fig[1, 2],
                   xlabel = L"Noise Level $\varepsilon$",
                   ylabel = "",
                   xscale = log10,
                   yscale = log10,
                   limits = (nothing, (1e-5, 1e3)),
                   xticks = ([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6], [L"10^{-1}", L"10^{-2}", L"10^{-3}", L"10^{-4}", L"10^{-5}", L"10^{-6}"]),
                   title = "Single Rank-$s Basis")

        # Plot selected methods for both cases
        for (ax, results) in [(ax1, results1), (ax2, results2)]
            for (linestyle, method_name) in [(:solid, "ttrand_orthogonal_blk"), (:dash, "stta_orthogonal_blk")]
                for block_rks in block_rks_list
                    full_key = method_name*"$(block_rks)"
                    if !haskey(results, full_key)
                        continue
                    end

                    data = results[full_key]
                    med = Float64.(data["median_error"])
                    err_low = Float64.(data["q25_error"])
                    err_high = Float64.(data["q75_error"])

                    color = colors_blk[block_rks]
                    marker = markers_blk[block_rks]
                    
                    band!(ax, ε_list, err_low, err_high,
                            color=(color, 0.3))
                    scatterlines!(ax, ε_list, med,
                            color=color,
                            linewidth=2,
                            linestyle=linestyle,
                            marker=marker,
                            markersize=5)
                end
            end

            # Add reference lines
            lines!(ax, ε_list, ε_list, label=L"$\varepsilon$ (reference)", color=:gray, linestyle=:dashdot, linewidth=1)
        end
        
        # Link the y-axes for consistent scaling
        linkaxes!(ax1, ax2)
        
        # Create custom legend elements
        line_elements = [
            LineElement(color = :black, linestyle = nothing, linewidth = 2),
            LineElement(color = :black, linestyle = :dot, linewidth = 2)
        ]
        line_labels = [LaTeXString("RandOrth"), LaTeXString("STTA")]
        
        
        ref_elements = [LineElement(color = :gray, linestyle = :dashdot, linewidth = 1)]
        ref_labels = [L"$\varepsilon$ (reference)"]
        
        # Combine all elements
        all_elements = [line_elements..., color_elements..., ref_elements...]
        all_labels = [line_labels..., color_labels..., ref_labels...]
        
        # Add horizontal legend below both subplots
        Legend(fig[2, 1:2], all_elements, all_labels,
               orientation = :horizontal, 
               tellheight = true, 
               framevisible = true,
               halign = :center,
               nbanks = 1,  # Single row
               padding = (2, 2, 2, 2),
               colgap = 8,
               labelsize = 9)

        display(fig)
        
        # Save the combined plot
        save("$dir/plots/combined_comparison_N$(results1["N"])_d$(results1["d"]).pdf", fig)
        println("Combined plot saved as PDF")
    end
end

"""
Run both experiments and create combined plot
"""
function run_randomized_rounding_experiments(; force_rerun = false)
    println("="^70)
    println("RANDOMIZED ROUNDING COMPARISON EXPERIMENTS")
    println("="^70)

    N = 50
    d = 4
    base_rank = 16
    block_rks_list = [1, 4, 8, 16]     # Number of blocks P = base_rank/block_rks => embedding dimension = base_rank
    perturbation_strengths = 10.0.^(-1:-1:-6)#10.0.^(-6:-1)
    perturbation_rank = 50
    n_realizations = 100
    dir = "out/randomized_rounding"

    # Run experiment 1: 8 summands of rank 1
    println("\n--- EXPERIMENT 1: 16 summands of rank 1 ---")
    results_16x1 = synthetic_experiment(
        N = N,
        d = d,
        base_rank = base_rank,
        n_summands = 16,
        perturbation_strengths = perturbation_strengths,
        perturbation_rank = perturbation_rank,
        block_rks_list = block_rks_list,
        n_realizations = n_realizations,
        force_rerun = force_rerun,
        dir = dir
    )
    
    # Run experiment 2: 1 summand of rank 8
    println("\n--- EXPERIMENT 2: 1 summand of rank 16 ---")
    results_1x16 = synthetic_experiment(
        N = N,
        d = d,
        base_rank = base_rank,
        n_summands = 1,
        perturbation_strengths = perturbation_strengths,
        perturbation_rank = perturbation_rank,
        block_rks_list = block_rks_list,
        n_realizations = n_realizations,
        force_rerun = force_rerun,
        dir = dir
    )

    # Create combined plot
    println("\n--- Creating combined comparison plot ---")
    create_combined_plot(results_16x1, results_1x16, dir=dir)

    println("\n" * "="^70)
    println("EXPERIMENTS COMPLETED")
    println("="^70)

    return (results_16x1, results_1x16)
end

# Run experiments automatically when file is included or executed
println("Running randomized rounding comparison experiments...")
global randomized_rounding_results = run_randomized_rounding_experiments()
