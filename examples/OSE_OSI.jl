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
    base_ranks = [1, 15],               # Ranks for generating subspace vectors (1 for rank-1, 15 for high-rank)
    P_list = [1, 4, 8],                 # Number of blocks to test
    n_realizations = 10,                # Number of realizations
    seed = 1234,
    test_types = [Float64],
    force_rerun = false,                # Force rerun even if results exist
    dir = "out/block_rank_experiments"
)
    results = Dict{String,Any}()
    for μ_rk in base_ranks
        # Check if results already exist
        mkpath(dir)
        filename = "$dir/ose_scaling_vs_dimension_mu$(μ)_rank-$(μ_rk).json"

        if !force_rerun && isfile(filename)
            println("=== Scenario 1: Injectivity vs Dimension N ===")
            println("Results already exist at: $filename")
            println("Loading existing results... (use force_rerun=true to recompute)")

            # Load and return existing results
            if isempty(results)
                results = JSON3.read(read(filename, String), Dict{String,Any})
            else
                tmp = JSON3.read(read(filename, String), Dict{String,Any})
                for key in keys(tmp)
                    results[key] = tmp[key]
                end
            end

            # Convert arrays to proper format (JSON converts them to nested structures)
            N_list_loaded = collect(results["N_list"])
            P_list_loaded = collect(results["P_list"])
            μ_loaded = results["μ"]

            # Convert nested arrays to matrices for each approach
            for key in keys(results)
                if key ∉ ["N_list", "μ", "d", "P_list", "n_realizations"]
                    approach_data = results[key]
                    for stat_key in ["median_injectivity", "q25_injectivity", "q75_injectivity",
                         "median_dilation", "q25_dilation", "q75_dilation"]
                        if haskey(approach_data, stat_key)
                            # Convert to matrix
                            arr = approach_data[stat_key]
                            approach_data[stat_key] = reshape(arr, :, length(P_list))
                        end
                    end
                end
            end
        else
            println("=== Scenario 1: Injectivity vs Dimension N ===")
            println("Fixed subspace size μ = $μ")
            println("Subspace spanned rank: $μ_rk")
            println("Dimension N values: $N_list")
            println("Number of blocks: $P_list")
            println("Physical dimension per core: $d")
            println("Realizations: $n_realizations")
            println()

            results["N_list"] = N_list
            results["μ"] = μ
            results["d"] = d
            results["P_list"] = P_list
            results["n_realizations"] = n_realizations

            flush(stdout) # <--- Force the text to appear in the Slurm output file

            # Test all 4 combinations
            for T in test_types
                type_name = T == Float64 ? "Float64" : "ComplexF64"

                for orthogonal in [true, false]
                    orth_name = orthogonal ? "orthogonal" : "non_orthogonal"
                    approach_name = "$(type_name)_$(orth_name)_rank-$(μ_rk)"

                    println("\n--- Testing $approach_name ---")
                    flush(stdout) # <--- Force the text to appear in the Slurm output file

                    # Storage: [N_index, block_rks_index, realization]
                    injectivity_data = zeros(length(N_list), length(P_list), n_realizations)
                    dilation_data = zeros(length(N_list), length(P_list), n_realizations)

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
                        for (i_p, P_i) in enumerate(P_list)
                            print("    P=$P_i: ")
                            flush(stdout) # <--- Force the text to appear in the Slurm output file

                            P = ones(Int, N+1)
                            P[2:N+1] .= P_i

                            # Get sketch dimension
                            _, sketch_rks = tt_recursive_sketch(T, X[1], μ;
                                                                orthogonal=orthogonal, reverse=false,
                                                                block_rks=2*N, p=P, seed=seed)
                            sketch_dim = sketch_rks[N+1]

                            for real = 1:n_realizations
                                # Build sketch matrix
                                Ω_matrix = zeros(T, sketch_dim, μ)
                                sketch_seed = seed + real + 1000*i_p + 10000*i_N

                                for j = 1:μ
                                    W, _ = tt_recursive_sketch(T, X[j], μ;
                                                            orthogonal=orthogonal, reverse=false,
                                                            seed=sketch_seed, block_rks=2*N, p=P)
                                    Ω_matrix[:, j] = W[N+1]'
                                end

                                # Apply orthogonalization and compute SVD
                                Ω_orth = Ω_matrix * C
                                σ = svdvals(Ω_orth)
                                injectivity_data[i_N, i_p, real] = σ[end]^2
                                dilation_data[i_N, i_p, real] = σ[1]^2

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
                                                for i in 1:length(N_list), j in 1:length(P_list)],
                        "q25_injectivity" => [quantile(injectivity_data[i, j, :], 0.25)
                                            for i in 1:length(N_list), j in 1:length(P_list)],
                        "q75_injectivity" => [quantile(injectivity_data[i, j, :], 0.75)
                                            for i in 1:length(N_list), j in 1:length(P_list)],



                        "median_dilation" => [median(dilation_data[i, j, :])
                                                for i in 1:length(N_list), j in 1:length(P_list)],
                        "q25_dilation" => [quantile(dilation_data[i, j, :], 0.25)
                                            for i in 1:length(N_list), j in 1:length(P_list)],
                        "q75_dilation" => [quantile(dilation_data[i, j, :], 0.75)
                                            for i in 1:length(N_list), j in 1:length(P_list)]
                    )
                end
            end

            # Save results after each base rank to ensure progress is not lost
            open(io -> JSON3.write(io, results, allow_inf=true), filename, "w")
            println("Results saved to: $filename")
            flush(stdout) # <--- Force the text to appear in the Slurm output file
        end
    end
    return results
end


"""
Create combined side-by-side plot comparing orthogonal vs non-orthogonal approaches
"""
function create_combined_scaling_plots(results1, N_list, P_list, μ_fixed, base_ranks; dir = "out/block_rank_experiments")
    mkpath("$dir/plots")

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
            P_list = Int.(P_list)

            # Color scheme for block ranks - extended for rank 16
            colors_P = Dict(1 => :blue, 2 => :orange, 4 => :green, 8 => :red)
            markers_P = Dict(1 => :circle, 2 => :rect, 4 => :diamond, 8 => :utriangle)

            #colors_P = Dict(5 => :blue, 10 => :orange, 15 => :green, 20 => :red)
            
            #markers_P = Dict(5 => :circle, 10 => :rect, 15 => :diamond, 20 => :utriangle)
            #colors_P = Dict(22 => :blue, 30 => :orange, 44 => :green, 88 => :red)
            #markers_P = Dict(22 => :circle, 30 => :rect, 44 => :diamond, 88 => :utriangle)

            fig = Figure(size = (700, 700))

            ax1 = Axis(fig[1, 1],
                    xlabel = L"Dimension $d$",
                    ylabel = L"Injectivity $\sigma^2_{\mathrm{min}}$",
                    #yscale = log10,
                    #limits = (nothing, (1e-7, 1e0)),
                    #yticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                    #         [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}"]),
                    title = LaTeXString("Injectivity vs d (r = $μ_fixed)"))

            ax3 = Axis(fig[1, 2],
                    xlabel = L"Dimension $d$",
                    ylabel = L"Dilation $\sigma^2_{\mathrm{max}}$",
                    #yscale = log10,
                    #limits = (nothing, (1e-2, 1e2)),
                    #yticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                    #         [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}"]),
                    title = LaTeXString("Dilation vs d (r = $μ_fixed)"))

            # Link axis for consistent scaling
            linkxaxes!(ax1, ax3)

            # Plot both orthogonal and non-orthogonal for scenario 1
            for (orth_flag, linestyle) in [(true, :solid), (false, :dash)]
                data_key = orth_flag ? "Float64_orthogonal" : "Float64_non_orthogonal"
                data1 = results1[data_key*"_rank-$(μ_rk)"]
                med1 = Float64.(data1["median_injectivity"])
                q25_1 = Float64.(data1["q25_injectivity"])
                q75_1 = Float64.(data1["q75_injectivity"])


                med3 = Float64.(data1["median_dilation"])
                q25_3 = Float64.(data1["q25_dilation"])
                q75_3 = Float64.(data1["q75_dilation"])


                for (i_p, P) in enumerate(P_list)
                    color = colors_P[P]
                    marker = markers_P[P]

                    # Extract data for this block rank across all N values
                    med_curve = med1[:, i_p]
                    err_low = q25_1[:, i_p]
                    err_high = q75_1[:, i_p]


                    med3_curve = med3[:, i_p]
                    err3_low = q25_3[:, i_p]
                    err3_high = q75_3[:, i_p]

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

            # Create custom legend with two rows
            # Row 1: Block ranks with colors
            color_elements = [
                LineElement(color = :blue, linewidth = 2),
                LineElement(color = :orange, linewidth = 2),
                LineElement(color = :green, linewidth = 2),
                LineElement(color = :red, linewidth = 2)
            ]
            color_labels = [L"P = 1", L"P = 2", L"P = 4", L"P = 8"]
            #color_labels = [L"P = 5", L"P = 10", L"P = 15", L"P = 20"]
            #color_labels = [L"P = 22", L"P = 30", L"P = 44", L"P = 88"]

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
            Legend(fig[2, 1:2], all_elements, all_labels,
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
            save("$dir/plots/ose_combined_scaling_comparison_rank-$μ_rk.pdf", fig)
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
    flush(stdout) # <--- Force the text to appear in the Slurm output file

    N_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    μ = 16
    ε = 1/2
    δ = 0.05
    P = ceil(Int, (μ + log(μ/δ)) / ε^2)
    #P_list = [ceil(Int, P/i) for i in 4:-1:1]
    #P_list = [5, 10, 15, 20]
    P_list = [1, 2, 4, 8]
    n_realizations = 100
    base_ranks = [1, 10]
    dir = "out/block_rank_experiments"

    # Scenario 1: Injectivity vs Dimension N
    results = injectivity_vs_dimension(
        N_list = N_list,
        μ = μ,
        base_ranks = base_ranks,
        P_list = P_list,
        n_realizations = n_realizations,
        force_rerun = force_rerun,
        dir = dir
    )

    println("\n" * "="^70)

    # Save results before plotting
    println("\n--- Saving experimental results ---")
    mkpath("out/block_rank_experiments")
    
    # Create combined side-by-side plot
    println("\n--- Creating combined comparison plot ---")
    flush(stdout) # <--- Force the text to appear in the Slurm output file
    create_combined_scaling_plots(results, 
                                  N_list, 
                                  P_list, 
                                  μ,
                                  base_ranks)

    println("\n" * "="^70)
    println("ALL EXPERIMENTS COMPLETED")
    println("="^70)

    return results
end

# Run experiments automatically when file is included or executed
println("Running all scaling experiments...")
global block_rank_scaling_experiments_results = run_all_scaling_experiments()
