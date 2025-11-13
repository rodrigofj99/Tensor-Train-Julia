using Revise
using TensorTrains
using LinearAlgebra
using Statistics
using JSON3
using Random
using Plots
using Printf

"""
Scaling Experiments for Forward Sketching

Studies how injectivity scales with:
1. Dimension N (for fixed subspace size μ)
2. Subspace size μ (for fixed dimension N)

For each scenario, creates 4 subplots comparing:
- Float64 Orthogonal
- Float64 Non-Orthogonal
- ComplexF64 Orthogonal
- ComplexF64 Non-Orthogonal

All subspaces are spanned by rank-1 vectors.
"""

"""
Scenario 1: Injectivity vs Dimension N (fixed subspace size μ)
"""
function injectivity_vs_dimension(;
    N_list = [5, 10, 15, 20, 25, 30],  # Dimension values to test
    d = 4,                              # Physical dimension per core
    μ = 20,                             # Fixed subspace size
    block_rks_list = [1, 2, 4, 8, 16], # Block ranks to test
    n_realizations = 10,                # Number of realizations
    seed = 1234,
    test_types = [Float64, ComplexF64],
    force_rerun = false                 # Force rerun even if results exist
)
    # Check if results already exist
    mkpath("out/block_rank_experiments")
    filename = "out/block_rank_experiments/scaling_vs_dimension_mu$(μ).json"

    if !force_rerun && isfile(filename)
        println("=== Scenario 1: Injectivity vs Dimension N ===")
        println("Results already exist at: $filename")
        println("Loading existing results... (use force_rerun=true to recompute)")

        # Load and return existing results
        results = JSON3.read(read(filename, String), Dict{String,Any})

        # Convert arrays to proper format (JSON converts them to nested structures)
        N_list_loaded = collect(results["N_list"])
        block_rks_list_loaded = collect(results["block_rks_list"])
        μ_loaded = results["μ"]

        # Convert nested arrays to matrices for each approach
        for key in keys(results)
            if key ∉ ["N_list", "μ", "d", "block_rks_list", "n_realizations"]
                approach_data = results[key]
                for stat_key in ["median_injectivity", "q25_injectivity", "q75_injectivity"]
                    if haskey(approach_data, stat_key)
                        # Convert to matrix
                        arr = approach_data[stat_key]
                        approach_data[stat_key] = reshape(arr, :, length(block_rks_list))
                    end
                end
            end
        end

        # Recreate plots from existing data
        println("Recreating plots from existing data...")
        create_dimension_plots(results, N_list_loaded, block_rks_list_loaded, μ_loaded)

        return results
    end

    println("=== Scenario 1: Injectivity vs Dimension N ===")
    println("Fixed subspace size μ = $μ")
    println("Dimension N values: $N_list")
    println("Block ranks: $block_rks_list")
    println("Physical dimension per core: $d")
    println("Realizations: $n_realizations")
    println()

    results = Dict{String,Any}()
    results["N_list"] = N_list
    results["μ"] = μ
    results["d"] = d
    results["block_rks_list"] = block_rks_list
    results["n_realizations"] = n_realizations

    # Test all 4 combinations
    for T in test_types
        type_name = T == Float64 ? "Float64" : "ComplexF64"

        for orthogonal in [true, false]
            orth_name = orthogonal ? "orthogonal" : "non_orthogonal"
            approach_name = "$(type_name)_$(orth_name)"

            println("\n--- Testing $approach_name ---")

            # Storage: [N_index, block_rks_index, realization]
            injectivity_data = zeros(length(N_list), length(block_rks_list), n_realizations)
            dilation_data = zeros(length(N_list), length(block_rks_list), n_realizations)

            for (i_N, N) in enumerate(N_list)
                println("  N = $N:")
                dims = ntuple(i -> d, N)

                # Generate rank-1 subspace
                X = Vector{TTvector{T,N}}(undef, μ)
                for j = 1:μ
                    rks_1 = ones(Int, N+1)
                    X[j] = rand_tt(T, dims, rks_1; orthogonal=true, normalise=false, seed=seed + j)
                    X[j] = X[j] / norm(X[j])
                end

                # Compute Cholesky orthogonalization
                A = zeros(T, μ, μ)
                for j1 = 1:μ
                    A[j1,j1] = dot(X[j1], X[j1])
                    for j2 = j1+1:μ
                        A[j1,j2] = dot(X[j1], X[j2])
                    end
                end
                A = Hermitian(A, :L)
                C = inv(cholesky(A).U)

                # Test each block rank
                for (i_blk, block_rks) in enumerate(block_rks_list)
                    print("    block_rks=$block_rks: ")

                    # Get sketch dimension
                    _, sketch_rks = tt_recursive_sketch(T, X[1], μ;
                                                        orthogonal=orthogonal, reverse=false,
                                                        block_rks=block_rks, seed=seed)
                    sketch_dim = sketch_rks[N+1]

                    for real = 1:n_realizations
                        # Build sketch matrix
                        Ω_matrix = zeros(T, sketch_dim, μ)
                        sketch_seed = seed + real + 1000*i_blk + 10000*i_N

                        for j = 1:μ
                            W, _ = tt_recursive_sketch(T, X[j], μ;
                                                      orthogonal=orthogonal, reverse=false,
                                                      seed=sketch_seed, block_rks=block_rks)
                            Ω_matrix[:, j] = W[N+1]'
                        end

                        # Apply orthogonalization and compute SVD
                        Ω_orth = Ω_matrix * C
                        σ = svdvals(Ω_orth)
                        injectivity_data[i_N, i_blk, real] = σ[end]^2
                        dilation_data[i_N, i_blk, real] = σ[1]^2

                        print(".")
                    end
                    println()
                end
            end

            # Compute statistics
            results[approach_name] = Dict(
                "injectivity" => injectivity_data,
                "dilation" => dilation_data,
                "median_injectivity" => [median(injectivity_data[i, j, :])
                                        for i in 1:length(N_list), j in 1:length(block_rks_list)],
                "q25_injectivity" => [quantile(injectivity_data[i, j, :], 0.25)
                                     for i in 1:length(N_list), j in 1:length(block_rks_list)],
                "q75_injectivity" => [quantile(injectivity_data[i, j, :], 0.75)
                                     for i in 1:length(N_list), j in 1:length(block_rks_list)]
            )
        end
    end

    # Create plots
    println("\n--- Creating plots ---")
    create_dimension_plots(results, N_list, block_rks_list, μ)

    # Save results
    mkpath("out/block_rank_experiments")
    filename = "out/block_rank_experiments/scaling_vs_dimension_mu$(μ).json"
    open(io -> JSON3.write(io, results, allow_inf=true), filename, "w")
    println("Results saved to: $filename")

    return results
end

"""
Scenario 2: Injectivity vs Subspace Size μ (fixed dimension N)
"""
function injectivity_vs_subspace_size(;
    N = 50,                             # Fixed dimension
    d = 4,                              # Physical dimension per core
    μ_list = [10, 20, 30, 40, 50],     # Subspace sizes to test
    block_rks_list = [1, 2, 4, 8, 16], # Block ranks to test
    n_realizations = 10,                # Number of realizations
    seed = 1234,
    test_types = [Float64, ComplexF64],
    force_rerun = false                 # Force rerun even if results exist
)
    # Check if results already exist
    mkpath("out/block_rank_experiments")
    filename = "out/block_rank_experiments/scaling_vs_subspace_N$(N).json"

    if !force_rerun && isfile(filename)
        println("=== Scenario 2: Injectivity vs Subspace Size μ ===")
        println("Results already exist at: $filename")
        println("Loading existing results... (use force_rerun=true to recompute)")

        # Load and return existing results
        results = JSON3.read(read(filename, String), Dict{String,Any})
        # Convert arrays to proper format (JSON converts them to nested structures)
        μ_list_loaded = collect(results["μ_list"])
        block_rks_list_loaded = collect(results["block_rks_list"])
        N_loaded = results["N"]

        # Convert nested arrays to matrices for each approach
        for key in keys(results)
            if key ∉ ["μ_list", "N", "d", "block_rks_list", "n_realizations"]
                approach_data = results[key]
                for stat_key in ["median_injectivity", "q25_injectivity", "q75_injectivity"]
                    if haskey(approach_data, stat_key)
                        # Convert to matrix
                        arr = approach_data[stat_key]
                        approach_data[stat_key] = reshape(arr, :, length(block_rks_list))
                    end
                end
            end
        end

        # Recreate plots from existing data
        println("Recreating plots from existing data...")
        create_subspace_plots(results, μ_list_loaded, block_rks_list_loaded, N_loaded)

        return results
    end

    println("=== Scenario 2: Injectivity vs Subspace Size μ ===")
    println("Fixed dimension N = $N")
    println("Subspace size μ values: $μ_list")
    println("Block ranks: $block_rks_list")
    println("Physical dimension per core: $d")
    println("Realizations: $n_realizations")
    println()

    dims = ntuple(i -> d, N)

    results = Dict{String,Any}()
    results["N"] = N
    results["d"] = d
    results["μ_list"] = μ_list
    results["block_rks_list"] = block_rks_list
    results["n_realizations"] = n_realizations

    # Test all 4 combinations
    for T in test_types
        type_name = T == Float64 ? "Float64" : "ComplexF64"

        for orthogonal in [true, false]
            orth_name = orthogonal ? "orthogonal" : "non_orthogonal"
            approach_name = "$(type_name)_$(orth_name)"

            println("\n--- Testing $approach_name ---")

            # Storage: [μ_index, block_rks_index, realization]
            injectivity_data = zeros(length(μ_list), length(block_rks_list), n_realizations)
            dilation_data = zeros(length(μ_list), length(block_rks_list), n_realizations)

            for (i_μ, μ) in enumerate(μ_list)
                println("  μ = $μ:")

                # Generate rank-1 subspace
                X = Vector{TTvector{T,N}}(undef, μ)
                for j = 1:μ
                    rks_1 = ones(Int, N+1)
                    X[j] = rand_tt(T, dims, rks_1; orthogonal=true, normalise=false, seed=seed + j)
                    X[j] = X[j] / norm(X[j])
                end

                # Compute Cholesky orthogonalization
                A = zeros(T, μ, μ)
                for j1 = 1:μ
                    A[j1,j1] = dot(X[j1], X[j1])
                    for j2 = j1+1:μ
                        A[j1,j2] = dot(X[j1], X[j2])
                    end
                end
                A = Hermitian(A, :L)
                C = inv(cholesky(A).U)

                # Test each block rank
                for (i_blk, block_rks) in enumerate(block_rks_list)
                    print("    block_rks=$block_rks: ")

                    # Get sketch dimension
                    _, sketch_rks = tt_recursive_sketch(T, X[1], μ;
                                                        orthogonal=orthogonal, reverse=false,
                                                        block_rks=block_rks, seed=seed)
                    sketch_dim = sketch_rks[N+1]

                    for real = 1:n_realizations
                        # Build sketch matrix
                        Ω_matrix = zeros(T, sketch_dim, μ)
                        sketch_seed = seed + real + 1000*i_blk + 10000*i_μ

                        for j = 1:μ
                            W, _ = tt_recursive_sketch(T, X[j], μ;
                                                      orthogonal=orthogonal, reverse=false,
                                                      seed=sketch_seed, block_rks=block_rks)
                            Ω_matrix[:, j] = W[N+1]'
                        end

                        # Apply orthogonalization and compute SVD
                        Ω_orth = Ω_matrix * C
                        σ = svdvals(Ω_orth)
                        injectivity_data[i_μ, i_blk, real] = σ[end]^2
                        dilation_data[i_μ, i_blk, real] = σ[1]^2

                        print(".")
                    end
                    println()
                end
            end

            # Compute statistics
            results[approach_name] = Dict(
                "injectivity" => injectivity_data,
                "dilation" => dilation_data,
                "median_injectivity" => [median(injectivity_data[i, j, :])
                                        for i in 1:length(μ_list), j in 1:length(block_rks_list)],
                "q25_injectivity" => [quantile(injectivity_data[i, j, :], 0.25)
                                     for i in 1:length(μ_list), j in 1:length(block_rks_list)],
                "q75_injectivity" => [quantile(injectivity_data[i, j, :], 0.75)
                                     for i in 1:length(μ_list), j in 1:length(block_rks_list)]
            )
        end
    end

    # Create plots
    println("\n--- Creating plots ---")
    create_subspace_plots(results, μ_list, block_rks_list, N)

    # Save results
    mkpath("out/block_rank_experiments")
    filename = "out/block_rank_experiments/scaling_vs_subspace_N$(N).json"
    open(io -> JSON3.write(io, results, allow_inf=true), filename, "w")
    println("Results saved to: $filename")

    return results
end

"""
Create 4-subplot figure for Scenario 1 (injectivity vs dimension N)
"""
function create_dimension_plots(results, N_list, block_rks_list, μ)
    mkpath("out/block_rank_experiments/plots")

    # Define the 4 approaches
    approaches = [
        ("Float64_orthogonal", "Real Orthogonal"),
        ("Float64_non_orthogonal", "Real Non-Orthogonal"),
        ("ComplexF64_orthogonal", "Complex Orthogonal"),
        ("ComplexF64_non_orthogonal", "Complex Non-Orthogonal")
    ]

    # Color scheme for block ranks
    colors = [:blue, :red, :green, :orange, :purple]
    markers = [:circle, :square, :diamond, :utriangle, :dtriangle]

    # Create 2×2 subplot layout
    plots = []

    for (approach_key, approach_title) in approaches
        data = results[approach_key]
        med = data["median_injectivity"]
        q25 = data["q25_injectivity"]
        q75 = data["q75_injectivity"]

        p = plot(xlabel="Dimension N", ylabel="Median Injectivity (σ²ₘᵢₙ)",
                 title=approach_title,
                 yscale=:log10,
                 ylims=(1e-6, 1), yticks=10.0.^(-6:0),
                 legend=:bottomleft,
                 size=(400, 350))

        # Plot each block rank as a separate series
        for (i_blk, block_rks) in enumerate(block_rks_list)
            # Extract data for this block rank across all N values
            med_curve = med[:, i_blk]
            err_low = med_curve .- q25[:, i_blk]
            err_high = q75[:, i_blk] .- med_curve

            plot!(p, N_list, med_curve,
                  yerror=(err_low, err_high),
                  label=( block_rks==1 ? "Khatri-Rao" : "Block ranks=$block_rks"),
                  marker=markers[i_blk],
                  color=colors[i_blk],
                  linewidth=2,
                  markersize=5,
                  markerstrokewidth=1.5)
        end

        push!(plots, p)
    end

    # Combine into 2×2 layout
    p_combined = plot(plots[1], plots[2], plots[3], plots[4],
                      layout=(2, 2),
                      size=(1000, 800),
                      plot_title="Injectivity vs Dimension N (μ=$μ)",
                      dpi=150)

    display(p_combined)
    savefig(p_combined, "out/block_rank_experiments/plots/scaling_vs_dimension_mu$(μ).png")
    println("Dimension scaling plot saved")
end

"""
Create 4-subplot figure for Scenario 2 (injectivity vs subspace size μ)
"""
function create_subspace_plots(results, μ_list, block_rks_list, N)
    mkpath("out/block_rank_experiments/plots")

    # Define the 4 approaches
    approaches = [
        ("Float64_orthogonal", "Real Orthogonal"),
        ("Float64_non_orthogonal", "Real Non-Orthogonal"),
        ("ComplexF64_orthogonal", "Complex Orthogonal"),
        ("ComplexF64_non_orthogonal", "Complex Non-Orthogonal")
    ]

    # Color scheme for block ranks
    colors = [:blue, :red, :green, :orange, :purple]
    markers = [:circle, :square, :diamond, :utriangle, :dtriangle]

    # Create 2×2 subplot layout
    plots = []

    for (approach_key, approach_title) in approaches
        data = results[approach_key]
        med = data["median_injectivity"]
        q25 = data["q25_injectivity"]
        q75 = data["q75_injectivity"]

        p = plot(xlabel="Subspace Size μ", ylabel="Median Injectivity (σ²ₘᵢₙ)",
                 title=approach_title,
                 xscale=:log2,
                 yscale=:log10,
                 ylims=(1e-9, 1e-1), yticks=10.0.^(-9:-1),
                 legend=:bottom,
                 size=(400, 350))

        # Plot each block rank as a separate series
        for (i_blk, block_rks) in enumerate(block_rks_list)
            # Extract data for this block rank across all μ values
            med_curve = med[:, i_blk]
            err_low = med_curve .- q25[:, i_blk]
            err_high = q75[:, i_blk] .- med_curve

            plot!(p, μ_list, med_curve,
                  yerror=(err_low, err_high),
                  label=( block_rks==1 ? "Khatri-Rao" : "Block ranks=$block_rks"),
                  marker=markers[i_blk],
                  color=colors[i_blk],
                  linewidth=2,
                  markersize=5,
                  markerstrokewidth=1.5)
        end

        push!(plots, p)
    end

    # Combine into 2×2 layout
    p_combined = plot(plots[1], plots[2], plots[3], plots[4],
                      layout=(2, 2),
                      size=(1000, 800),
                      plot_title="Injectivity vs Subspace Size μ (N=$N)",
                      dpi=150)

    display(p_combined)
    savefig(p_combined, "out/block_rank_experiments/plots/scaling_vs_subspace_N$(N).png")
    println("Subspace scaling plot saved")
end

"""
Run both scenarios
"""
function run_all_scaling_experiments(; force_rerun = false)
    println("="^70)
    println("RUNNING ALL SCALING EXPERIMENTS")
    println("="^70)

    # Scenario 1: Injectivity vs Dimension N
    results1 = injectivity_vs_dimension(
        N_list = [5, 10, 15, 20, 25, 30],
        μ = 20,
        block_rks_list = [1, 2, 4, 8, 16],
        n_realizations = 100,
        force_rerun = force_rerun
    )

    println("\n" * "="^70)

    # Scenario 2: Injectivity vs Subspace Size μ
    results2 = injectivity_vs_subspace_size(
        N = 50,
        μ_list = [8, 16, 32, 64, 128],
        block_rks_list = [1, 2, 4, 8, 16],
        n_realizations = 100,
        force_rerun = force_rerun
    )

    println("\n" * "="^70)
    println("ALL EXPERIMENTS COMPLETED")
    println("="^70)

    return results1, results2
end

# Run experiments automatically when file is included or executed
println("Running all scaling experiments...")
global block_rank_scaling_experiments_results = run_all_scaling_experiments()
