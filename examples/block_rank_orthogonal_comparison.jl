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
Orthogonal vs Non-Orthogonal Forward Sketch Comparison

Compares orthogonal and non-orthogonal forward sketching approaches across different block ranks
and number types (Float64 vs ComplexF64), measuring both injectivity quality and computational performance.

Generates two plots:
1. Median injectivity (+ error bars) vs block rank
2. Median injectivity vs average timing (performance-quality tradeoff)
"""
function orthogonal_comparison_study(;
    N = 20,                          # Number of cores
    dims = ntuple(i -> 4, 20),       # Physical dimensions (all = 4)
    subspace_dim = 32,               # Dimension of test subspace
    rank_1_count = 32,               # Number of rank-1 vectors (uniform rank-1 subspace)
    rank_high = 1,                   # Not used for uniform rank-1
    block_rks_list = [1, 2, 4, 8, 16, 32, 64],  # Block ranks to test
    n_realizations = 100,            # Number of independent realizations
    seed = 1234,                     # Random seed
    test_types = [Float64, ComplexF64],  # Number types to test
    force_rerun = false              # Force re-running even if data exists
)
    println("=== Orthogonal vs Non-Orthogonal Sketch Comparison ===")
    println("N = $N, dims = $(dims[1]) (uniform), subspace_dim = $subspace_dim")
    println("Block ranks: $block_rks_list")
    println("Realizations: $n_realizations")
    println("Types to test: $test_types")
    println()

    # Check if data already exists
    mkpath("out/block_rank_experiments")
    filename = "out/block_rank_experiments/orthogonal_comparison_N$(N).json"

    if !force_rerun && isfile(filename)
        println("Loading existing data from $filename...")
        results = JSON3.read(read(filename, String))

        # Convert back to proper types
        results = Dict{String,Any}(string(k) => v for (k,v) in pairs(results))

        # Verify data matches parameters
        if results["N"] == N && results["subspace_dim"] == subspace_dim &&
           results["block_rks_list"] == block_rks_list
            println("Data matches current parameters. Creating plots...")
            create_comparison_plots(results, block_rks_list, N)
            println("Plots created successfully!")
            return results
        else
            println("Parameters don't match. Re-running experiments...")
        end
    end

    println("Running experiments...")
    # Storage for results
    results = Dict{String,Any}()
    results["dims"] = dims
    results["N"] = N
    results["subspace_dim"] = subspace_dim
    results["block_rks_list"] = block_rks_list
    results["n_realizations"] = n_realizations
    results["test_types"] = [string(T) for T in test_types]

    # Test all combinations: (orthogonal, type)
    for T in test_types
        type_name = T == Float64 ? "Float64" : "ComplexF64"

        # Create uniform rank-1 subspace for this type
        X = Vector{TTvector{T,N}}(undef, subspace_dim)

        println("\n=== Type: $type_name ===")
        println("Generating rank-1 test subspace...")
        for i = 1:subspace_dim
            rks_1 = ones(Int, N+1)
            X[i] = rand_tt(T, dims, rks_1; orthogonal=true, normalise=false, seed=seed + i)
            X[i] = X[i] / norm(X[i])
        end

        # Compute Gram matrix and orthogonalization factor
        println("Computing Cholesky orthogonalization...")
        A = zeros(T, subspace_dim, subspace_dim)
        for i = 1:subspace_dim
            A[i,i] = dot(X[i], X[i])
            for j = i+1:subspace_dim
                A[i,j] = dot(X[i], X[j])
            end
        end
        A = Hermitian(A, :L)
        C = inv(cholesky(A).U)

        for orthogonal in [true, false]
            approach_name = orthogonal ? "$(type_name)_orthogonal" : "$(type_name)_non_orthogonal"
            println("\n--- Testing $approach_name forward sketching ---")

            injectivity = zeros(length(block_rks_list), n_realizations)
            dilation = zeros(length(block_rks_list), n_realizations)
            timings = zeros(length(block_rks_list), n_realizations)
            sketch_dims = zeros(Int, length(block_rks_list))

            for (i_blk, block_rks) in enumerate(block_rks_list)
                println("    Block rank $block_rks:")

                # Get sketch dimension
                _, sketch_rks = tt_recursive_sketch(T, X[1], subspace_dim;
                                                    orthogonal=orthogonal, reverse=false,
                                                    block_rks=block_rks, seed=seed)
                sketch_dim = sketch_rks[N+1]
                sketch_dims[i_blk] = sketch_dim
                println("      Sketch dimension: $sketch_dim")

                # Warm-up run to eliminate compilation overhead
                print("      Warming up...")
                W, _ = tt_recursive_sketch(T, X[1], subspace_dim;
                                              orthogonal=orthogonal, reverse=false,
                                              seed=seed + 9999, block_rks=block_rks,
                                              timer=TimerOutput())
                println(" done")

                for real = 1:n_realizations
                    # Create timer for this realization
                    to = TimerOutput()

                    # Build sketch matrix with timing
                    Ω_matrix = zeros(T, sketch_dim, subspace_dim)
                    sketch_seed = seed + real + 1000*i_blk

                    for j = 1:subspace_dim
                        # Sketch each vector with timer
                        W, _ = tt_recursive_sketch(T, X[j], subspace_dim;
                                                  orthogonal=orthogonal, reverse=false,
                                                  seed=sketch_seed, block_rks=block_rks,
                                                  timer=to)

                        # Extract sketch vector (forward: use W[N+1])
                        Ω_matrix[:, j] = W[N+1]'
                    end

                    # Extract total time from TimerOutput (in seconds)
                    timings[i_blk, real] = TimerOutputs.tottime(to) / 1e9

                    # Apply orthogonalization and compute SVD
                    Ω_orth = Ω_matrix * C
                    σ = svdvals(Ω_orth)
                    injectivity[i_blk, real] = σ[end]^2
                    dilation[i_blk, real] = σ[1]^2

                    print(".")
                end
                println(" (avg: $(round(mean(timings[i_blk,:]), digits=3))s)")
            end

            # Compute statistics
            results[approach_name] = Dict(
                "sketch_dims" => sketch_dims,
                "injectivity" => injectivity,
                "dilation" => dilation,
                "timings" => timings,
                "median_injectivity" => [median(injectivity[i,:]) for i in 1:length(block_rks_list)],
                "q25_injectivity" => [quantile(injectivity[i,:], 0.25) for i in 1:length(block_rks_list)],
                "q75_injectivity" => [quantile(injectivity[i,:], 0.75) for i in 1:length(block_rks_list)],
                "median_dilation" => [median(dilation[i,:]) for i in 1:length(block_rks_list)],
                "mean_timing" => [mean(timings[i,:]) for i in 1:length(block_rks_list)],
                "std_timing" => [std(timings[i,:]) for i in 1:length(block_rks_list)]
            )
        end
    end

    # Print summary
    println("\n=== Results Summary ===")
    approach_names = []
    for T in test_types
        type_name = T == Float64 ? "Float64" : "ComplexF64"
        for orth in ["orthogonal", "non_orthogonal"]
            push!(approach_names, "$(type_name)_$(orth)")
        end
    end

    for approach_name in approach_names
        println("\n$approach_name:")
        println("  Block ranks:        ", block_rks_list)
        println("  Sketch dims:        ", results[approach_name]["sketch_dims"])
        println("  Median injectivity: ", [@sprintf("%.4e", x) for x in results[approach_name]["median_injectivity"]])
        println("  Mean timing (s):    ", [@sprintf("%.3f", x) for x in results[approach_name]["mean_timing"]])
    end

    # Generate plots
    println("\n--- Creating plots ---")
    create_comparison_plots(results, block_rks_list, N)

    # Save results
    mkpath("out/block_rank_experiments")
    filename = "out/block_rank_experiments/orthogonal_comparison_N$(N).json"
    open(io -> JSON3.write(io, results, allow_inf=true), filename, "w")
    println("\nResults saved to: $filename")

    return results
end

"""
Create comparison plots for all 4 combinations (orthogonal/non-orthogonal × Float64/ComplexF64)
"""
function create_comparison_plots(results, block_rks_list, N)
    mkpath("out/block_rank_experiments/plots")

    # Define all approaches with colors and markers
    approaches = [
        ("Float64_orthogonal", "F64 Orth", :blue, :circle),
        ("Float64_non_orthogonal", "F64 NonOrth", :cyan, :square),
        ("ComplexF64_orthogonal", "C64 Orth", :red, :circle),
        ("ComplexF64_non_orthogonal", "C64 NonOrth", :orange, :square)
    ]

    # Plot 1: Median injectivity vs block rank with error bars
    p1 = plot(xlabel="Block Rank", ylabel="Median Injectivity (σ²ₘᵢₙ)",
              title="Sketch Quality Comparison",
              yscale=:log10, xscale=:log2,
              ylims=(1e-5, 1), yticks=10.0.^(-5:0),
              legend=:bottomright, size=(800, 600), dpi=150)

    for (key, label, color, marker) in approaches
        data = results[key]
        med = data["median_injectivity"]
        err_low = med .- data["q25_injectivity"]
        err_high = data["q75_injectivity"] .- med

        plot!(p1, block_rks_list, med,
              yerror=(err_low, err_high),
              label=label, marker=marker, linewidth=2,
              markersize=5, color=color, markerstrokewidth=1.5)
    end

    display(p1)
    savefig(p1, "out/block_rank_experiments/plots/injectivity_vs_block_rank_N$(N).png")
    println("Plot 1 saved: injectivity_vs_block_rank_N$(N).png")

    # Plot 2: Median injectivity vs average timing (performance-quality tradeoff)
    p2 = plot(xlabel="Average Timing (seconds)", ylabel="Median Injectivity (σ²ₘᵢₙ)",
              title="Performance-Quality Tradeoff",
              yscale=:log10, xscale=:log10,
              ylims=(1e-5, 1), yticks=10.0.^(-5:0),
              legend=:bottomleft, size=(800, 600), dpi=150)

    for (key, label, color, marker) in approaches
        data = results[key]
        med = data["median_injectivity"]
        time = data["mean_timing"]

        plot!(p2, time, med,
              label=label, marker=marker, linewidth=2, linestyle=:dash,
              markersize=6, color=color, markerstrokewidth=1.5)
    end

    display(p2)
    savefig(p2, "out/block_rank_experiments/plots/injectivity_vs_timing_N$(N).png")
    println("Plot 2 saved: injectivity_vs_timing_N$(N).png")

    # Plot 3: Direct timing comparison
    p3 = plot(xlabel="Block Rank", ylabel="Average Timing (seconds)",
              title="Computational Cost Comparison",
              xscale=:log2, yscale=:log10,
              legend=:topleft, size=(800, 600), dpi=150)

    for (key, label, color, marker) in approaches
        data = results[key]
        time = data["mean_timing"]
        time_std = data["std_timing"]

        plot!(p3, block_rks_list, time,
              yerror=time_std,
              label=label, marker=marker, linewidth=2,
              markersize=5, color=color, markerstrokewidth=1.5)
    end

    display(p3)
    savefig(p3, "out/block_rank_experiments/plots/timing_vs_block_rank_N$(N).png")
    println("Plot 3 saved: timing_vs_block_rank_N$(N).png")

    # Plot 4: Orthogonal vs Non-orthogonal for each type separately
    for type_name in ["Float64", "ComplexF64"]
        p4 = plot(xlabel="Block Rank", ylabel="Median Injectivity (σ²ₘᵢₙ)",
                  title="$type_name: Orthogonal vs Non-Orthogonal",
                  yscale=:log10, xscale=:log2,
                  ylims=(1e-5, 1), yticks=10.0.^(-5:0),
                  legend=:bottomright, size=(700, 500), dpi=150)

        orth_key = "$(type_name)_orthogonal"
        non_orth_key = "$(type_name)_non_orthogonal"

        # Orthogonal
        data_orth = results[orth_key]
        med_orth = data_orth["median_injectivity"]
        err_low_orth = med_orth .- data_orth["q25_injectivity"]
        err_high_orth = data_orth["q75_injectivity"] .- med_orth

        plot!(p4, block_rks_list, med_orth,
              yerror=(err_low_orth, err_high_orth),
              label="Orthogonal", marker=:circle, linewidth=2.5,
              markersize=6, color=:blue, markerstrokewidth=2)

        # Non-orthogonal
        data_non = results[non_orth_key]
        med_non = data_non["median_injectivity"]
        err_low_non = med_non .- data_non["q25_injectivity"]
        err_high_non = data_non["q75_injectivity"] .- med_non

        plot!(p4, block_rks_list, med_non,
              yerror=(err_low_non, err_high_non),
              label="Non-Orthogonal", marker=:square, linewidth=2.5,
              markersize=6, color=:red, markerstrokewidth=2)

        display(p4)
        savefig(p4, "out/block_rank_experiments/plots/$(type_name)_comparison_N$(N).png")
        println("Plot 4 ($type_name) saved")
    end

    println("\nAll plots created successfully!")
end

# Run the study automatically when file is included or executed
if !isdefined(Main, :block_rank_orthogonal_comparison_results)
    println("Running orthogonal comparison study...")
    global block_rank_orthogonal_comparison_results = orthogonal_comparison_study()
end
