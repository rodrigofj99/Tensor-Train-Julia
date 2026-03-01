using Revise
using TensorTrains
using LinearAlgebra
using Statistics
using JSON3
using Random
using CairoMakie
using LaTeXStrings
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
"""

"""
Scenario 1: Injectivity vs Dimension N (fixed subspace size μ)
"""
function injectivity_vs_dimension(;
    N_list = [5, 10, 15, 20, 25, 30],   # Dimension values to test
    d = 4,                              # Physical dimension per core
    μ = 32,                             # Fixed subspace size
    base_ranks = [1, 15],                  # Ranks for generating subspace vectors (1 for rank-1, 15 for high-rank)
    block_rks_list = [1, 4, 8],         # Block ranks to test
    n_realizations = 10,                # Number of realizations
    seed = 1234,
    test_types = [Float64],
    force_rerun = false                 # Force rerun even if results exist
)
    results = Dict{String,Any}()
    for μ_rk in base_ranks
        # Check if results already exist
        mkpath("out/block_rank_experiments")
        filename = "out/block_rank_experiments/scaling_vs_dimension_mu$(μ)_rank-$(μ_rk).json"

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
                    for stat_key in ["median_injectivity", "q25_injectivity", "q75_injectivity",
                         "median_dilation", "q25_dilation", "q75_dilation"]
                        if haskey(approach_data, stat_key)
                            # Convert to matrix
                            arr = approach_data[stat_key]
                            approach_data[stat_key] = reshape(arr, :, length(block_rks_list))
                        end
                    end
                end
            end

            return results
        end

        println("=== Scenario 1: Injectivity vs Dimension N ===")
        println("Fixed subspace size μ = $μ")
        println("Subspace spanned rank: $μ_rk")
        println("Dimension N values: $N_list")
        println("Block ranks: $block_rks_list")
        println("Physical dimension per core: $d")
        println("Realizations: $n_realizations")
        println()

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
                approach_name = "$(type_name)_$(orth_name)_rank-$(μ_rk)"

                println("\n--- Testing $approach_name ---")

                # Storage: [N_index, block_rks_index, realization]
                injectivity_data = zeros(length(N_list), length(block_rks_list), n_realizations)
                dilation_data = zeros(length(N_list), length(block_rks_list), n_realizations)

                for (i_N, N) in enumerate(N_list)
                    println("  N = $N:")
                    dims = ntuple(i -> d, N)

                    # Generate rank-μ_rk subspace
                    X = Vector{TTvector{T,N}}(undef, μ)
                    for j = 1:μ
                        rks_1 = ntuple(i -> μ_rk, N+1)
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
                        _, sketch_rks = tt_recursive_sketch(T, X[1], 2μ;
                                                            orthogonal=orthogonal, reverse=false,
                                                            block_rks=block_rks, seed=seed)
                        sketch_dim = sketch_rks[N+1]

                        for real = 1:n_realizations
                            # Build sketch matrix
                            Ω_matrix = zeros(T, sketch_dim, μ)
                            sketch_seed = seed + real + 1000*i_blk + 10000*i_N

                            for j = 1:μ
                                W, _ = tt_recursive_sketch(T, X[j], 2μ;
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
                                        for i in 1:length(N_list), j in 1:length(block_rks_list)],



                    "median_dilation" => [median(dilation_data[i, j, :])
                                            for i in 1:length(N_list), j in 1:length(block_rks_list)],
                    "q25_dilation" => [quantile(dilation_data[i, j, :], 0.25)
                                        for i in 1:length(N_list), j in 1:length(block_rks_list)],
                    "q75_dilation" => [quantile(dilation_data[i, j, :], 0.75)
                                        for i in 1:length(N_list), j in 1:length(block_rks_list)]
                )
            end
        end

        # Note: Individual plots handled by combined plotting function

        # Save results
        mkpath("out/block_rank_experiments")
        filename = "out/block_rank_experiments/scaling_vs_dimension_mu$(μ)_rank-$(μ_rk).json"
        open(io -> JSON3.write(io, results, allow_inf=true), filename, "w")
        println("Results saved to: $filename")

    end

    return results
end

"""
Scenario 2: Injectivity vs Subspace Size μ (fixed dimension N)
"""
function injectivity_vs_subspace_size(;
    N = 50,                             # Fixed dimension
    d = 4,                              # Physical dimension per core
    μ_list = [10, 20, 30, 40, 50],      # Subspace sizes to test
    base_ranks = [1, 15],                  # Ranks for generating subspace vectors (1 for rank-1, 15 for high-rank)
    block_rks_list = [1, 4, 8],         # Block ranks to test
    n_realizations = 10,                # Number of realizations
    seed = 1234,
    test_types = [Float64],
    force_rerun = false                 # Force rerun even if results exist
)
    results = Dict{String,Any}()
    for μ_rk in base_ranks
        # Check if results already exist
        mkpath("out/block_rank_experiments")
        filename = "out/block_rank_experiments/scaling_vs_subspace_N$(N)_rank-$(μ_rk).json"

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
                    for stat_key in ["median_injectivity", "q25_injectivity", "q75_injectivity",
                         "median_dilation", "q25_dilation", "q75_dilation"]
                        if haskey(approach_data, stat_key)
                            # Convert to matrix
                            arr = approach_data[stat_key]
                            approach_data[stat_key] = reshape(arr, :, length(block_rks_list))
                        end
                    end
                end
            end

            # Note: Individual plots replaced by combined plotting function

            return results
        end

        println("=== Scenario 2: Injectivity vs Subspace Size μ ===")
        println("Fixed dimension N = $N")
        println("Subspace size μ values: $μ_list")
        println("Subspace spanned rank: $μ_rk")
        println("Block ranks: $block_rks_list")
        println("Physical dimension per core: $d")
        println("Realizations: $n_realizations")
        println()

        dims = ntuple(i -> d, N)

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
                approach_name = "$(type_name)_$(orth_name)_rank-$(μ_rk)"

                println("\n--- Testing $approach_name ---")

                # Storage: [μ_index, block_rks_index, realization]
                injectivity_data = zeros(length(μ_list), length(block_rks_list), n_realizations)
                dilation_data = zeros(length(μ_list), length(block_rks_list), n_realizations)

                for (i_μ, μ) in enumerate(μ_list)
                    println("  μ = $μ:")

                    # Generate rank-μ_rk subspace
                    X = Vector{TTvector{T,N}}(undef, μ)
                    for j = 1:μ
                        rks_1 = ntuple(i -> μ_rk, N+1)
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
                        _, sketch_rks = tt_recursive_sketch(T, X[1], 2μ;
                                                            orthogonal=orthogonal, reverse=false,
                                                            block_rks=block_rks, seed=seed)
                        sketch_dim = sketch_rks[N+1]

                        for real = 1:n_realizations
                            # Build sketch matrix
                            Ω_matrix = zeros(T, sketch_dim, μ)
                            sketch_seed = seed + real + 1000*i_blk + 10000*i_μ

                            for j = 1:μ
                                W, _ = tt_recursive_sketch(T, X[j], 2μ;
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
                                        for i in 1:length(μ_list), j in 1:length(block_rks_list)],



                    "median_dilation" => [median(dilation_data[i, j, :])
                                            for i in 1:length(μ_list), j in 1:length(block_rks_list)],
                    "q25_dilation" => [quantile(dilation_data[i, j, :], 0.25)
                                        for i in 1:length(μ_list), j in 1:length(block_rks_list)],
                    "q75_dilation" => [quantile(dilation_data[i, j, :], 0.75)
                                        for i in 1:length(μ_list), j in 1:length(block_rks_list)]
                )
            end
        end

        # Note: Individual plots handled by combined plotting function

        # Save results
        mkpath("out/block_rank_experiments")
        filename = "out/block_rank_experiments/scaling_vs_subspace_N$(N)_rank-$(μ_rk).json"
        open(io -> JSON3.write(io, results, allow_inf=true), filename, "w")
        println("Results saved to: $filename")

    end
    return results
end

"""
Create combined side-by-side plot comparing orthogonal vs non-orthogonal approaches
"""
function create_combined_scaling_plots(results1, results2, N_list, μ_list, block_rks_list, μ_fixed, N_fixed, base_ranks)
    mkpath("out/block_rank_experiments/plots")

    # Configure CairoMakie for publication-quality plots
    CairoMakie.activate!(type = "pdf")

    # Define consistent styling theme (same as randomized rounding)
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
            rowgap = 3,
            colgap = 8,
            labelsize = 9
        )
    )

    for μ_rk in base_ranks
        with_theme(PLOT_THEME) do
            # Ensure proper types for plotting
            N_list = Int.(N_list)
            μ_list = Int.(μ_list)
            block_rks_list = Int.(block_rks_list)

            # Color scheme for block ranks - extended for rank 16
            colors_blk = Dict(1 => :blue, 4 => :orange, 16 => :green, 32 => :red)
            markers_blk = Dict(1 => :circle, 4 => :rect, 16 => :diamond, 32 => :utriangle)

            ###colors_blk = Dict(1 => :blue, 4 => :orange)
            ###markers_blk = Dict(1 => :circle, 4 => :rect)

            #fig = Figure(size = (624, 300))  # 6.5" wide, 4.5" tall (72 DPI equivalent)
            fig = Figure()#size = (624, 624))

            ax1 = Axis(fig[1, 1],
                    xlabel = L"Dimension $d$",
                    ylabel = L"Injectivity $\sigma^2_{\mathrm{min}}$",
                    yscale = log10,
                    limits = (nothing, (1e-7, 1e0)),
                    #yticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                    #         [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}"]),
                    title = LaTeXString("Injectivity vs d (r = $μ_fixed)"))

            #ax2 = Axis(fig[1, 3],
            ax2 = Axis(fig[1, 2],
                    xlabel = L"Subspace Size $r$",
                    #ylabel = L"Injectivity $\sigma^2_{\mathrm{min}}$",
                    xscale = log2,
                    yscale = log10,
                    limits = (nothing, (1e-7, 1e0)),
                    xticks = ([8, 16, 32, 64, 128], ["8", "16", "32", "64", "128"]),
                    #xticks = (μ_list, string.(μ_list)),
                    #yticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                    #         [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}"]),
                    title = LaTeXString("Injectivity vs r (d = $N_fixed)"))

            ax3 = Axis(fig[2, 1],
                    xlabel = L"Dimension $d$",
                    ylabel = L"Dilation $\sigma^2_{\mathrm{max}}$",
                    yscale = log10,
                    limits = (nothing, (1e-2, 1e2)),
                    #yticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                    #         [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}"]),
                    title = LaTeXString("Dilation vs d (r = $μ_fixed)"))

            #ax4 = Axis(fig[1, 4],
            ax4 = Axis(fig[2, 2],
                    xlabel = L"Subspace Size $r$",
                    #ylabel = L"Dilation $\sigma^2_{\mathrm{max}}$",
                    xscale = log2,
                    yscale = log10,
                    limits = (nothing, (1e-2, 1e2)),
                    xticks = ([8, 16, 32, 64, 128], ["8", "16", "32", "64", "128"]),
                    #xticks = (μ_list, string.(μ_list)),
                    #yticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                    #         [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}"]),
                    title = LaTeXString("Dilation vs r (d = $N_fixed)"))

            # Link axis for consistent scaling
            linkyaxes!(ax1, ax2)
            linkyaxes!(ax3, ax4)
            linkxaxes!(ax1, ax3)
            linkxaxes!(ax2, ax4)

            # Plot both orthogonal and non-orthogonal for scenario 1: N vs injectivity
            for (orth_flag, linestyle) in [(true, :solid), (false, :dash)]
                data_key = orth_flag ? "Float64_orthogonal" : "Float64_non_orthogonal"
                data1 = results1[data_key*"_rank-$(μ_rk)"]
                med1 = Float64.(data1["median_injectivity"])
                q25_1 = Float64.(data1["q25_injectivity"])
                q75_1 = Float64.(data1["q75_injectivity"])


                med3 = Float64.(data1["median_dilation"])
                q25_3 = Float64.(data1["q25_dilation"])
                q75_3 = Float64.(data1["q75_dilation"])


                for (i_blk, block_rks) in enumerate(block_rks_list)
                    color = colors_blk[block_rks]
                    marker = markers_blk[block_rks]

                    # Extract data for this block rank across all N values
                    med_curve = med1[:, i_blk]
                    err_low = q25_1[:, i_blk]
                    err_high = q75_1[:, i_blk]


                    med3_curve = med3[:, i_blk]
                    err3_low = q25_3[:, i_blk]
                    err3_high = q75_3[:, i_blk]

                    band!(ax1, N_list, err_low, err_high, color=(color, 0.2))
                    scatterlines!(ax1, N_list, med_curve,
                                color=color,
                                linewidth=2,
                                linestyle=linestyle,
                                marker=marker,
                                markersize=6)


                    band!(ax3, N_list, err3_low, err3_high, color=(color, 0.2))
                    scatterlines!(ax3, N_list, med3_curve,
                                color=color,
                                linewidth=2,
                                linestyle=linestyle,
                                marker=marker,
                                markersize=6)
                end
            end

            # Plot both orthogonal and non-orthogonal for scenario 2: μ vs injectivity
            for (orth_flag, linestyle) in [(true, :solid), (false, :dash)]
                data_key = orth_flag ? "Float64_orthogonal" : "Float64_non_orthogonal"
                data2 = results2[data_key*"_rank-$(μ_rk)"]
                med2 = Float64.(data2["median_injectivity"])
                q25_2 = Float64.(data2["q25_injectivity"])
                q75_2 = Float64.(data2["q75_injectivity"])


                med4 = Float64.(data2["median_dilation"])
                q25_4 = Float64.(data2["q25_dilation"])
                q75_4 = Float64.(data2["q75_dilation"])

                for (i_blk, block_rks) in enumerate(block_rks_list)
                    color = colors_blk[block_rks]
                    marker = markers_blk[block_rks]

                    # Extract data for this block rank across all μ values
                    med_curve = med2[:, i_blk]
                    err_low = q25_2[:, i_blk]
                    err_high = q75_2[:, i_blk]


                    med4_curve = med4[:, i_blk]
                    err4_low = q25_4[:, i_blk]
                    err4_high = q75_4[:, i_blk]

                    band!(ax2, μ_list, err_low, err_high, color=(color, 0.2))
                    scatterlines!(ax2, μ_list, med_curve,
                                color=color,
                                linewidth=2,
                                linestyle=linestyle,
                                marker=marker,
                                markersize=6)



                    band!(ax4, μ_list, err4_low, err4_high, color=(color, 0.2))
                    scatterlines!(ax4, μ_list, med4_curve,
                                color=color,
                                linewidth=2,
                                linestyle=linestyle,
                                marker=marker,
                                markersize=6)
                end
            end

            # Create custom legend with two rows
            # Row 1: Block ranks with colors
            color_elements = [
                LineElement(color = :blue, linewidth = 2),
                LineElement(color = :orange, linewidth = 2),
                LineElement(color = :green, linewidth = 2),
                LineElement(color = :red, linewidth = 2)
            ]
            color_labels = [LaTeXString("R=1(Khatri-Rao)"), L"R = 4", L"R = 16", L"R = 32"]

            # Row 2: Orthogonal vs Non-orthogonal line styles
            style_elements = [
                LineElement(color = :black, linewidth = 2, linestyle = :dash),
                LineElement(color = :black, linewidth = 2, linestyle = :solid)
            ]
            style_labels = ["Gaussian i.i.d", "Orthogonal"]

            # Combine elements
            all_elements = [color_elements..., style_elements...]
            all_labels = [color_labels..., style_labels...]

            # Add horizontal legend below both subplots
            #Legend(fig[2, 1:4], all_elements, all_labels,
            Legend(fig[3, 1:2], all_elements, all_labels,
                orientation = :horizontal,
                tellheight = true,
                framevisible = true,
                halign = :center,
                nbanks = 1,  # Single row
                padding = (4, 4, 2, 2),
                colgap = 8,
                labelsize = 9)

            display(fig)
            sleep(2)

            # Save the combined plot
            save("out/block_rank_experiments/plots/combined_scaling_comparison_rank-$μ_rk.pdf", fig)
            println("Combined scaling plot saved as PDF")
        end
    end
end


"""
Run both scenarios comparing orthogonal and non-orthogonal approaches
"""
function run_all_scaling_experiments(; force_rerun = false)
    println("="^70)
    println("RUNNING BLOCK RANK SCALING EXPERIMENTS")
    println("="^70)

    N_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    N = 20
    μ_list = [8, 16, 32, 64, 128]
    μ = 16
    block_rks_list = [1, 4, 16, 32]
    n_realizations = 100
    base_ranks = [1, 32]

    # Scenario 1: Injectivity vs Dimension N
    results1 = injectivity_vs_dimension(
        N_list = N_list,
        μ = μ,
        base_ranks = base_ranks,
        block_rks_list = block_rks_list,
        n_realizations = n_realizations,
        force_rerun = force_rerun
    )

    println("\n" * "="^70)

    # Scenario 2: Injectivity vs Subspace Size μ
    results2 = injectivity_vs_subspace_size(
        N = N,
        μ_list = μ_list,
        base_ranks = base_ranks,
        block_rks_list = block_rks_list,
        n_realizations = n_realizations,
        force_rerun = force_rerun
    )

    # Save results before plotting
    println("\n--- Saving experimental results ---")
    mkpath("out/block_rank_experiments")
    
    # Create combined side-by-side plot
    println("\n--- Creating combined comparison plot ---")
    create_combined_scaling_plots(results1, results2, 
                                  N_list, 
                                  μ_list,
                                  block_rks_list, 
                                  μ,
                                  N,
                                  base_ranks)

    println("\n" * "="^70)
    println("ALL EXPERIMENTS COMPLETED")
    println("="^70)

    return results1, results2
end

# Run experiments automatically when file is included or executed
println("Running all scaling experiments...")
global block_rank_scaling_experiments_results = run_all_scaling_experiments()
