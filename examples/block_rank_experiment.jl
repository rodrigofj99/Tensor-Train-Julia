using Revise
using TensorTrains
using LinearAlgebra
using Statistics
using JSON3
using Random
using StaticArrays
using Plots
using Printf
using TimerOutputs

"""
Specialized OSI Study: N=5, dims=4, uniform ranks

Tests the specific case requested by the user:
- N=5 cores, all dimensions = 4
- Block ranks from 1 to 64
- Two subspace types: (1) uniform rank 1, (2) uniform rank 10
- Both forward and reverse sketching
"""
function osi_uniform_ranks_study()
    println("=== Specialized OSI Study: N=20, dims=4, Uniform Ranks ===")
    println("Testing block ranks 1-64 with forward and reverse sketching")
    println()
    
    N = 20
    dims = ntuple(i -> 4, N)  # All dimensions = 4
    block_rks_list = [1,2,4,8,16,32,64]
    n_realizations = 10
    
    results = Dict()
    
    # Test case 1: Uniform rank 1 subspace
    println("--- Case 1: Uniform Rank 1 Subspace ---")
    results["rank_1"] = osi_block_rank_study(ComplexF64,
        dims = dims,
        subspace_dim = 32,   # Use larger subspace for better OSI demonstration
        rank_1_count = 32,   # All rank-1 vectors
        rank_high = 1,      # Not used since all are rank-1
        block_rks_list = collect(block_rks_list),
        n_realizations = n_realizations,
        seed = 1234,
        test_forward = true,
        create_plots = true  # Disable individual plots to avoid layout issues
    )
    
    # println("\n" * "="^60 * "\n")
    
    # Test case 2: Uniform rank 10 subspace  
    println("--- Case 2: Uniform Rank 10 Subspace ---")
    results["rank_10"] = osi_block_rank_study(ComplexF64,
        dims = dims,
        subspace_dim = 32,   # Use larger subspace for better OSI demonstration
        rank_1_count = 0,   # No rank-1 vectors
        rank_high = 10,     # All rank-10 vectors
        block_rks_list = collect(block_rks_list),
        n_realizations = n_realizations,
        seed = 5678,        # Different seed for independence
        test_forward = true,
        create_plots = false  # Disable individual plots to avoid layout issues
    )
    
    # Create only comparative plots with better error handling
    println("\n--- Creating Comparative Analysis ---")
    create_comparative_plots_safe(results, block_rks_list, N)
    
    # Save combined results
    mkpath("out/block_rank_experiments")
    filename = "out/block_rank_experiments/osi_uniform_ranks_N$(N)_dims4.json"
    open(io -> JSON3.write(io, results, allow_inf=true), filename, "w")
    println("Combined results saved to: $filename")
    
    return results
end


"""
OSI Property Study with Block Rank Dependence

Studies the Orthogonal Sketch Invariance (OSI) property by examining how the lower
singular values of sketched subspaces depend on the block rank parameter.

This function creates a subspace spanned by TTvectors of different ranks, orthogonalizes
it using Cholesky decomposition, then studies how block rank affects the sketch quality.
"""
function osi_block_rank_study(::Type{T}=Float64;
    dims = (2,2,2,2,2),  # Tensor dimensions  
    subspace_dim = 20,   # Dimension of subspace to study
    rank_1_count = 10,   # Number of rank-1 TTvectors
    rank_high = 5,       # Rank for higher-rank TTvectors
    block_rks_list = [1,2,3,4,5],  # Block ranks to test
    n_realizations = 5,  # Number of random realizations
    seed = 1234,         # Random seed
    test_forward = true, # Test forward sketching in addition to reverse
    create_plots = true  # Create plots of the results
) where {T}
    N = length(dims)
    
    println("=== OSI Property Block Rank Study ===")
    println("Dimensions: $dims")
    println("Subspace dimension: $subspace_dim")
    println("Rank-1 vectors: $rank_1_count, High-rank vectors: $(subspace_dim - rank_1_count) (rank $rank_high)")
    println("Block ranks to test: $block_rks_list")
    println("Realizations: $n_realizations")
    println()
    
    # Create subspace X of dimension subspace_dim
    X = Vector{TTvector{T,N}}(undef, subspace_dim)
    
    # Generate rank-1 TTvectors
    for i = 1:rank_1_count
        rks_1 = ones(Int, N+1)  # Rank-1 for any N
        X[i] = rand_tt(T, dims, rks_1; orthogonal=true, normalise=false, seed=seed + i)
        X[i] = X[i] / norm(X[i])
    end
    
    # Generate higher-rank TTvectors
    for i = (rank_1_count+1):subspace_dim
        rks_high = [1, fill(rank_high, N-1)..., 1]
        # Ensure ranks don't exceed dimension limits
        for k=1:N
            if rks_high[k+1] > dims[k]*rks_high[k]
                rks_high[k+1] = dims[k]*rks_high[k]
            end
        end
        for k=N:-1:1
            if rks_high[k] > dims[k]*rks_high[k+1]
                rks_high[k] = dims[k]*rks_high[k+1]
            end
        end
        
        X[i] = rand_tt(T, dims, rks_high; orthogonal=true, normalise=false, seed=seed + i + 1000)
        X[i] = X[i] / norm(X[i])
    end
    
    # Compute Gram matrix and Cholesky factor for orthogonalization
    A = zeros(T, subspace_dim, subspace_dim)
    for i = 1:subspace_dim
        A[i,i] = dot(X[i], X[i])
        for j = i+1:subspace_dim
            A[i,j] = dot(X[i], X[j])
        end
    end
    A = Hermitian(A, :L)
    # Cholesky factor for orthogonalization: X*C is orthogonal
    C = inv(cholesky(A).U)
    
    # Results storage
    results = Dict{String,Any}()
    results["dims"] = dims
    results["subspace_dim"] = subspace_dim
    results["rank_1_count"] = rank_1_count
    results["rank_high"] = rank_high
    results["block_rks_list"] = block_rks_list
    results["n_realizations"] = n_realizations
    
    # Determine sketch directions to test
    directions = ["reverse"]
    if test_forward
        directions = ["reverse", "forward"]
    end
    
    # Storage for singular values (injectivity = smallest, dilation = largest)
    results_by_direction = Dict()
    
    for direction in directions
        println("\n--- Testing $direction sketching ---")
        reverse_sketch = (direction == "reverse")
        
        injectivity = zeros(length(block_rks_list), n_realizations)
        dilation = zeros(length(block_rks_list), n_realizations)
        sketch_dims = zeros(Int, length(block_rks_list))
        
        for (i_blk, block_rks) in enumerate(block_rks_list)
            to = TimerOutput()
            println("Testing block_rks = $block_rks")

            # First, get the actual sketch dimension by computing one sketch
            _, sketch_rks = tt_recursive_sketch(T, X[1], subspace_dim;
                                                           orthogonal=true, reverse=reverse_sketch,
                                                           block_rks=block_rks)
            if reverse_sketch
                sketch_dim = sketch_rks[1]
            else
                sketch_dim = sketch_rks[N+1]
            end
            sketch_dims[i_blk] = sketch_dim

            for real = 1:n_realizations
                # Create sketch matrix Ω
                Ω_matrix = zeros(T, sketch_dim, subspace_dim)

                # Generate sketch using recursive sketch with specified block rank
                sketch_seed = seed + real + 1000*i_blk
                for j = 1:subspace_dim
                    # Use tt_recursive_sketch with current block_rks parameter
                    W, _ = tt_recursive_sketch(T, X[j], subspace_dim; 
                                             orthogonal=true, reverse=reverse_sketch, 
                                             seed=sketch_seed,
                                             block_rks=block_rks, timer=to)
                    
                    # Extract the sketch vector from the appropriate matrix
                    if reverse_sketch
                        # For reverse: W[1] has size (X[j].ttv_rks[1], sketch_rks_actual[1])
                        # Since X[j].ttv_rks[1] = 1, this gives us a row vector
                        Ω_matrix[:, j] = W[1]'
                    else
                        # For forward: W[N+1] has size (X[j].ttv_rks[N+1], sketch_rks_actual[N+1])  
                        # Since X[j].ttv_rks[N+1] = 1, this gives us a row vector
                        Ω_matrix[:, j] = W[N+1]'
                    end
                end
                
                # Apply Cholesky orthogonalization: compute Ω * C
                Ω_orth = Ω_matrix * C
                
                # Compute SVD and extract singular values
                σ = svdvals(Ω_orth)
                injectivity[i_blk, real] = σ[end]^2    # Smallest singular value squared
                dilation[i_blk, real] = σ[1]^2        # Largest singular value squared
                
                print(".")
            end
            println()
            display(to)
        end
        
        # Store results for this direction
        results_by_direction[direction] = Dict(
            "sketch_dims" => sketch_dims,
            "injectivity" => injectivity,
            "dilation" => dilation,
            "mean_injectivity" => [mean(injectivity[i,:]) for i in 1:length(block_rks_list)],
            "std_injectivity" => [std(injectivity[i,:]) for i in 1:length(block_rks_list)],
            "median_injectivity" => [median(injectivity[i,:]) for i in 1:length(block_rks_list)],
            "mean_dilation" => [mean(dilation[i,:]) for i in 1:length(block_rks_list)],
            "std_dilation" => [std(dilation[i,:]) for i in 1:length(block_rks_list)],
            "median_dilation" => [median(dilation[i,:]) for i in 1:length(block_rks_list)]
        )
    end
    
    # Store results by direction
    for direction in directions
        results[direction] = results_by_direction[direction]
    end
    
    # Print summary for each direction
    println("\n=== OSI Results Summary ===")
    println("Block ranks: ", block_rks_list)
    for direction in directions
        println("\n--- $direction sketching ---")
        println("block rank:         [",["$(@sprintf "%6i" x), " for x in block_rks_list[1:end-1]]..., "$(@sprintf "%6i" block_rks_list[end])]")
        println("sketch dimension:   [",["$(@sprintf "%6i" x), " for x in results[direction]["sketch_dims"][1:end-1]]..., "$(@sprintf "%6i" results[direction]["sketch_dims"][end])]")
        println("Mean injectivity:   ", [round(x, digits=6) for x in results[direction]["mean_injectivity"]])
        println("Median injectivity: ", [round(x, digits=6) for x in results[direction]["median_injectivity"]])
        println("Std injectivity:    ", [round(x, digits=6) for x in results[direction]["std_injectivity"]])
        println("Mean dilation:      ", [round(x, digits=6) for x in results[direction]["mean_dilation"]])
        println("Median dilation:    ", [round(x, digits=6) for x in results[direction]["median_dilation"]])
        println("Std dilation:       ", [round(x, digits=6) for x in results[direction]["std_dilation"]])
    end

    
    # Create plots if requested
    if create_plots
        plots_list = []
        
        for direction in directions
            # Injectivity plot
            p1 = plot(block_rks_list, results[direction]["mean_injectivity"],
                     yerror=results[direction]["std_injectivity"],
                     label="$direction sketching",
                     xlabel="Block Rank",
                     ylabel="Injectivity (smallest σ²)",
                     title="OSI Injectivity vs Block Rank",
                     marker=:circle,
                     linewidth=2,
                     markersize=4,
                     yscale=:log10)
            
            # Dilation plot
            p2 = plot(block_rks_list, results[direction]["mean_dilation"],
                     yerror=results[direction]["std_dilation"],
                     label="$direction sketching",
                     xlabel="Block Rank",
                     ylabel="Dilation (largest σ²)",
                     title="OSI Dilation vs Block Rank",
                     marker=:circle,
                     linewidth=2,
                     markersize=4,
                     yscale=:log10)
            
            push!(plots_list, p1, p2)
        end
        
        # Combined plot
        if length(directions) > 1
            p_combined = plot()
            for direction in directions
                plot!(p_combined, block_rks_list, results[direction]["mean_injectivity"],
                     yerror=results[direction]["std_injectivity"],
                     label="$direction sketching",
                     xlabel="Block Rank",
                     ylabel="Injectivity (smallest σ²)",
                     title="OSI Injectivity: Forward vs Reverse Sketching",
                     marker=:circle,
                     linewidth=2,
                     markersize=4,
                     yscale=:log10)
            end
            push!(plots_list, p_combined)
        end
        
        # Display plots
        for (i, p) in enumerate(plots_list)
            display(p)
        end
        
        # Save plots
        mkpath("out/block_rank_experiments/plots")
        for (i, p) in enumerate(plots_list)
            plot_name = "out/block_rank_experiments/plots/osi_plot_$(i)_N=$(N)_sub=$(subspace_dim).png"
            savefig(p, plot_name)
            println("Plot saved: $plot_name")
        end
    end
    
    # Save results
    mkpath("out/block_rank_experiments")
    filename = "out/block_rank_experiments/osi_block_rank_study_N=$(N)_sub=$(subspace_dim).json"
    open(io -> JSON3.write(io, results, allow_inf=true), filename, "w")
    println("Results saved to: $filename")
    
    return results
end

"""
Create comparative plots between rank-1 and rank-10 subspaces with robust error handling
"""
function create_comparative_plots_safe(results, block_rks_list, N)
    mkpath("out/block_rank_experiments/plots")
    
    # Check what data is available
    has_rank_1 = haskey(results, "rank_1")
    has_rank_10 = haskey(results, "rank_10")
    
    if !has_rank_1 && !has_rank_10
        println("Warning: No results available for plotting")
        return
    end
    
    # Force backend and create plots one by one to avoid layout conflicts
    plots_created = 0
    
    # Simple plotting with minimal dependencies on layout engine
    try
        # Initialize plots backend
        gr()  # Use GR backend explicitly
        
        # Plot 1: Statistical equivalence test for rank-1
        if has_rank_1 && haskey(results["rank_1"], "reverse") && haskey(results["rank_1"], "forward")
            try
                reverse_data = results["rank_1"]["reverse"]["median_injectivity"]
                forward_data = results["rank_1"]["forward"]["median_injectivity"]
                
                # Create completely fresh plot
                plot() # Clear any existing plot state
                p1 = plot(block_rks_list, reverse_data, 
                         label="Reverse", marker=:circle, linewidth=2, color=:blue,
                         size=(600, 400), dpi=150, yscale=:log10)
                plot!(p1, block_rks_list, forward_data, 
                      label="Forward", marker=:square, linewidth=2, color=:red)
                xlabel!(p1, "Block Rank")
                ylabel!(p1, "Median Injectivity")
                title!(p1, "OSI Equivalence Test: Forward vs Reverse (Rank-1)")
                
                max_diff = maximum(abs.(reverse_data .- forward_data))
                println("Maximum absolute difference between forward/reverse (rank-1): $max_diff")
                
                plot_file = "out/block_rank_experiments/plots/equivalence_test_rank1_N$(N).png"
                savefig(p1, plot_file)
                println("Equivalence test plot saved: $plot_file")
                plots_created += 1
            catch e
                println("Warning: Could not create equivalence test plot for rank-1: $e")
            end
        end
        
        # Plot 2: Statistical equivalence test for rank-10
        if has_rank_10 && haskey(results["rank_10"], "reverse") && haskey(results["rank_10"], "forward")
            try
                reverse_data = results["rank_10"]["reverse"]["median_injectivity"] 
                forward_data = results["rank_10"]["forward"]["median_injectivity"]
                
                plot() # Clear plot state
                p2 = plot(block_rks_list, reverse_data,
                         label="Reverse", marker=:circle, linewidth=2, color=:blue,
                         size=(600, 400), dpi=150, yscale=:log10)
                plot!(p2, block_rks_list, forward_data,
                      label="Forward", marker=:square, linewidth=2, color=:red)
                xlabel!(p2, "Block Rank")
                ylabel!(p2, "Median Injectivity")
                title!(p2, "OSI Equivalence Test: Forward vs Reverse (Rank-10)")
                
                max_diff = maximum(abs.(reverse_data .- forward_data))
                println("Maximum absolute difference between forward/reverse (rank-10): $max_diff")
                
                plot_file = "out/block_rank_experiments/plots/equivalence_test_rank10_N$(N).png"
                savefig(p2, plot_file)
                println("Equivalence test plot saved: $plot_file")
                plots_created += 1
            catch e
                println("Warning: Could not create equivalence test plot for rank-10: $e")
            end
        end
        
        # Plot 3 & 4: Rank comparison plots
        if has_rank_1 && has_rank_10
            for direction in ["reverse", "forward"]
                if (haskey(results["rank_1"], direction) && haskey(results["rank_10"], direction))
                    try
                        plot() # Clear plot state
                        p3 = plot(block_rks_list, results["rank_1"][direction]["median_injectivity"],
                                 label="Uniform Rank 1", marker=:circle, linewidth=2, color=:blue,
                                 size=(600, 400), dpi=150, yscale=:log10)
                        plot!(p3, block_rks_list, results["rank_10"][direction]["median_injectivity"],
                              label="Uniform Rank 10", marker=:square, linewidth=2, color=:red)
                        xlabel!(p3, "Block Rank")
                        ylabel!(p3, "Median Injectivity")
                        title!(p3, "OSI Rank Comparison: $(uppercasefirst(direction)) Sketching")
                        
                        plot_file = "out/block_rank_experiments/plots/rank_comparison_$(direction)_N$(N).png"
                        savefig(p3, plot_file)
                        println("Rank comparison plot saved: $plot_file")
                        plots_created += 1
                    catch e
                        println("Warning: Could not create $direction rank comparison plot: $e")
                    end
                end
            end
        end
        
    catch e
        println("Warning: Overall plotting error: $e")
    end
    
    if plots_created == 0
        println("No plots could be created successfully")
    else
        println("Successfully created $plots_created comparative plots")
    end
end

"""
Create comparative plots between rank-1 and rank-10 subspaces (original function kept for compatibility)
"""
function create_comparative_plots(results, block_rks_list, N)
    mkpath("out/block_rank_experiments/plots")
    
    plots_list = []
    plot_names = []
    
    # Check what data is available
    has_rank_1 = haskey(results, "rank_1")
    has_rank_10 = haskey(results, "rank_10")
    
    # Only create comparison plots if we have both rank types
    if has_rank_1 && has_rank_10
        # Plot 1: Injectivity comparison - Reverse sketching
        if haskey(results["rank_1"], "reverse") && haskey(results["rank_10"], "reverse")
            try
                p1 = plot(xlabel="Block Rank", ylabel="Injectivity (smallest σ²)", 
                          title="OSI Injectivity: Rank-1 vs Rank-10 Subspaces (Reverse)")
                plot!(p1, block_rks_list, results["rank_1"]["reverse"]["mean_injectivity"],
                      yerror=results["rank_1"]["reverse"]["std_injectivity"],
                      label="Uniform Rank 1", marker=:circle, linewidth=2, color=:blue)
                plot!(p1, block_rks_list, results["rank_10"]["reverse"]["mean_injectivity"],
                      yerror=results["rank_10"]["reverse"]["std_injectivity"],
                      label="Uniform Rank 10", marker=:square, linewidth=2, color=:red)
                # Only use log scale if data spans more than one order of magnitude
                all_data = vcat(results["rank_1"]["reverse"]["mean_injectivity"], 
                               results["rank_10"]["reverse"]["mean_injectivity"])
                if maximum(all_data) / minimum(all_data) > 10
                    plot!(p1, yscale=:log10)
                end
                push!(plots_list, p1)
                push!(plot_names, "rank_comparison_reverse")
            catch e
                println("Warning: Could not create reverse comparison plot: $e")
            end
        end
        
        # Plot 2: Injectivity comparison - Forward sketching
        if haskey(results["rank_1"], "forward") && haskey(results["rank_10"], "forward")
            try
                p2 = plot(xlabel="Block Rank", ylabel="Injectivity (smallest σ²)",
                          title="OSI Injectivity: Rank-1 vs Rank-10 Subspaces (Forward)")
                plot!(p2, block_rks_list, results["rank_1"]["forward"]["mean_injectivity"],
                      yerror=results["rank_1"]["forward"]["std_injectivity"],
                      label="Uniform Rank 1", marker=:circle, linewidth=2, color=:blue)
                plot!(p2, block_rks_list, results["rank_10"]["forward"]["mean_injectivity"],
                      yerror=results["rank_10"]["forward"]["std_injectivity"],
                      label="Uniform Rank 10", marker=:square, linewidth=2, color=:red)
                # Only use log scale if data spans more than one order of magnitude
                all_data = vcat(results["rank_1"]["forward"]["mean_injectivity"], 
                               results["rank_10"]["forward"]["mean_injectivity"])
                if maximum(all_data) / minimum(all_data) > 10
                    plot!(p2, yscale=:log10)
                end
                push!(plots_list, p2)
                push!(plot_names, "rank_comparison_forward")
            catch e
                println("Warning: Could not create forward comparison plot: $e")
            end
        end
    end
    
    # Create direction comparison plots for each rank type available
    if has_rank_1 && haskey(results["rank_1"], "reverse") && haskey(results["rank_1"], "forward")
        try
            p3 = plot(xlabel="Block Rank", ylabel="Injectivity (smallest σ²)",
                      title="OSI Injectivity: Forward vs Reverse (Rank-1 Subspace)")
            plot!(p3, block_rks_list, results["rank_1"]["reverse"]["mean_injectivity"],
                  yerror=results["rank_1"]["reverse"]["std_injectivity"],
                  label="Reverse", marker=:circle, linewidth=2, color=:blue)
            plot!(p3, block_rks_list, results["rank_1"]["forward"]["mean_injectivity"],
                  yerror=results["rank_1"]["forward"]["std_injectivity"],
                  label="Forward", marker=:square, linewidth=2, color=:green)
            # Only use log scale if data spans more than one order of magnitude
            all_data = vcat(results["rank_1"]["reverse"]["mean_injectivity"], 
                           results["rank_1"]["forward"]["mean_injectivity"])
            if maximum(all_data) / minimum(all_data) > 10
                plot!(p3, yscale=:log10)
            end
            push!(plots_list, p3)
            push!(plot_names, "direction_comparison_rank1")
        catch e
            println("Warning: Could not create rank-1 direction comparison plot: $e")
        end
    end
    
    if has_rank_10 && haskey(results["rank_10"], "reverse") && haskey(results["rank_10"], "forward")
        try
            p4 = plot(xlabel="Block Rank", ylabel="Injectivity (smallest σ²)",
                      title="OSI Injectivity: Forward vs Reverse (Rank-10 Subspace)")
            plot!(p4, block_rks_list, results["rank_10"]["reverse"]["mean_injectivity"],
                  yerror=results["rank_10"]["reverse"]["std_injectivity"],
                  label="Reverse", marker=:circle, linewidth=2, color=:blue)
            plot!(p4, block_rks_list, results["rank_10"]["forward"]["mean_injectivity"],
                  yerror=results["rank_10"]["forward"]["std_injectivity"],
                  label="Forward", marker=:square, linewidth=2, color=:green)
            # Only use log scale if data spans more than one order of magnitude
            all_data = vcat(results["rank_10"]["reverse"]["mean_injectivity"], 
                           results["rank_10"]["forward"]["mean_injectivity"])
            if maximum(all_data) / minimum(all_data) > 10
                plot!(p4, yscale=:log10)
            end
            push!(plots_list, p4)
            push!(plot_names, "direction_comparison_rank10")
        catch e
            println("Warning: Could not create rank-10 direction comparison plot: $e")
        end
    end
    
    # Save plots that were created
    if length(plots_list) > 0
        for (i, (p, name)) in enumerate(zip(plots_list, plot_names))
            display(p)
            plot_file = "out/block_rank_experiments/plots/$(name)_N$(N).png"
            savefig(p, plot_file)
            println("Comparative plot saved: $plot_file")
        end
    else
        println("No comparative plots created - insufficient data available")
    end
end


"""
Quick OSI test with smaller dimensions to validate fixes
"""
function quick_osi_test()
    println("=== Quick OSI Test (N=3) ===")
    
    # Small test to validate fixes
    results = osi_block_rank_study(
        dims = (3,3,3),  # Small N for quick test
        subspace_dim = 3,
        rank_1_count = 2,
        rank_high = 2,
        sketch_dim = 6,
        block_rks_list = [1,2],  # Just 2 block ranks for quick test
        n_realizations = 50,     # More realizations for better statistics
        seed = 1234,
        test_forward = true,
        create_plots = false  # Disable plots for quick test
    )
    
    # Analyze results
    if haskey(results, "reverse") && haskey(results, "forward")
        reverse_data = results["reverse"]["mean_injectivity"]
        forward_data = results["forward"]["mean_injectivity"]
        reverse_std = results["reverse"]["std_injectivity"] 
        forward_std = results["forward"]["std_injectivity"]
        
        println("\nComparison of Forward vs Reverse Sketching:")
        println("Reverse: mean = $reverse_data, std = $reverse_std")
        println("Forward: mean = $forward_data, std = $forward_std")
        
        # Statistical significance test (rough)
        for i in 1:length(reverse_data)
            pooled_std = sqrt((reverse_std[i]^2 + forward_std[i]^2)/2)
            t_stat = abs(reverse_data[i] - forward_data[i]) / pooled_std * sqrt(25)  # rough t-test
            println("Block rank $(results["block_rks_list"][i]): diff = $(abs(reverse_data[i] - forward_data[i])), t-stat ≈ $t_stat")
        end
        
        println("\nNote: Forward and reverse sketching may have different distributions")
        println("This is not necessarily a bug - they sample different subspaces")
    end
    
    return results
end

# Run quick test if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    quick_osi_test()
end