using Revise
using TensorTrains
using LinearAlgebra
using Random
using Printf
using QuanticsTCI
using Statistics
using TimerOutputs
using Plots
using JSON3
using Dates

"""
QuanticsTCI Hadamard Product Sketch Benchmark

Tests structured sketching of Hadamard products using QuanticsTCI-derived tensor trains.
This follows the pattern from quantics_rounding_benchmark.jl but focuses on testing
the multi-TTvector Kronecker sketching approach for 3-term products.

The benchmark:
1. Uses QuanticsTCI to create tensor train representations of term1, term2, term3 functions
2. Tests ttrand_rounding((tt1, tt2, tt3), target_ranks; block_rks=...) 
3. Compares against exact computation: tt1 * tt2 * tt3 → tt_rounding(product, tol=1e-12)
4. Evaluates different block rank configurations: 1 (Khatri-Rao), 4 (small), 8 (large)
"""

include("quantics_example.jl")

"""
Generate QuanticsTCI tensor trains for the 3 test functions
"""
function generate_quantics_test_tensors(R::Int; tolerance=1e-10, seed=1234)
    Random.seed!(seed)
    
    println("=== Generating QuanticsTCI tensor trains (R=$R) ===")
    println("Grid size: $(2^R)³ = $(2^(3*R)) total points")
    println("TCI tolerance: $tolerance")
    
    # Create coordinate arrays for each dimension
    xvals = range(0.0, 1.0; length=2^R)
    yvals = range(0.0, 1.0; length=2^R)
    zvals = range(0.0, 1.0; length=2^R)
    
    # Create the three function terms
    f1(x, y, z) = term1(x, y, z, R)
    f2(x, y, z) = term2(x, y, z, R)
    f3(x, y, z) = term3(x, y, z, R)
    f = (f1,f2,f3)
    
    println("Starting QuanticsTCI interpolation for 3 terms...")
    
    # Interpolate each term separately
    interpolants = ntuple(j -> 
        quanticscrossinterpolate(
            Float64, 
            f[j], 
            [xvals, yvals, zvals]; 
            tolerance=tolerance,
            unfoldingscheme=:interleaved
        ), 3)
    
    qtt_interpolants = getindex.(interpolants, 1)
    ranks = getindex.(interpolants, 2)
    errors = getindex.(interpolants, 3)
    
    println("QuanticsTCI interpolation completed!")
    for i in 1:3
        println("  Term $i: ranks = $(ranks[i])")
    end
    
    # Convert to TensorTrains format
    println("Converting to TTvector format...")
    ttvectors = qtt_to_ttvector.(qtt_interpolants)
    
    println("Successfully created 3 TTvectors:")
    for i in 1:3
        ttv = ttvectors[i]
        println("  Term $i: dims = $(ttv.ttv_dims), ranks = $(ttv.ttv_rks)")
        println("          max rank = $(maximum(ttv.ttv_rks)), parameters = $(sum(prod(size(core)) for core in ttv.ttv_vec))")
    end
    
    return tuple(ttvectors...)
end

"""
Compute exact Hadamard product using deterministic compression
"""
function compute_exact_hadamard(ttvectors::NTuple{3,TTvector{T,N}}; tol=1e-12) where {T,N}
    println("Computing exact Hadamard product...")
    
    # Compute the 3-way product
    exact_product = ttvectors[1] * ttvectors[2] * ttvectors[3]
    println("  Raw product: max rank = $(maximum(exact_product.ttv_rks))")
    
    # Compress with high precision
    exact_compressed = tt_rounding(exact_product, tol=tol)
    println("  Compressed: max rank = $(maximum(exact_compressed.ttv_rks))")
    println("  Compression ratio: $(maximum(exact_product.ttv_rks))/$(maximum(exact_compressed.ttv_rks)) = $(@sprintf "%.2f" maximum(exact_product.ttv_rks)/maximum(exact_compressed.ttv_rks))")
    
    return exact_compressed
end

"""
Test structured sketching with different block rank configurations
"""
function benchmark_structured_sketching(
    ttvectors::NTuple{3,TTvector{T,N}}, 
    exact_solution::TTvector{T,N};
    target_ranks = [16,32,48,64],
    block_rks_options = [1, 4, 16],
    n_trials = 3,
    seed = 1234
) where {T,N}
    
    println("\n=== Structured Sketching Benchmark ===")
    println("Testing 3-term Hadamard product with structured sketching")
    println("Target ranks: $target_ranks")
    println("Block rank configurations: $block_rks_options")
    println("Number of trials: $n_trials")
    
    exact_norm = norm(exact_solution)
    println("Reference norm: $(@sprintf "%.6e" exact_norm)")
    
    results = Dict{String, Any}()
    
    for target_rank in target_ranks
        println("\n" * "="^60)
        println("Testing target_rank = $target_rank")
        println("="^60)
        
        # Create uniform target rank vector
        target_rks = [1; fill(target_rank, N-1); 1]
        println("Target ranks: $target_rks")
        
        rank_results = Dict{String, Any}("target_rank" => target_rank)
        
        # Warm-start: Run each configuration once to avoid precompilation overhead
        println("Warm-starting all configurations...")
        
        # Warm up deterministic algorithm
        try
            println("  Warming up tt_rounding (deterministic)...")
            product = ttvectors[1] * ttvectors[2] * ttvectors[3]
            tt_rounding(product; rmax=target_rank)
            println("    ✓ Deterministic warm-up complete")
        catch e
            println("    ✗ Deterministic warm-up failed: $e")
        end
        
        # Warm up randomized algorithms
        for block_rks in block_rks_options
            try
                println("  Warming up ttrand_rounding block_rks=$block_rks...")
                ttrand_rounding(ttvectors, target_rks; block_rks=block_rks, seed=seed)
                println("    ✓ Warm-up complete")
            catch e
                println("    ✗ Warm-up failed: $e")
            end
        end
        println("All warm-ups complete. Starting benchmark trials...\n")
        
        # Test deterministic algorithm first
        println("--- Deterministic tt_rounding ---")
        
        det_errors = Float64[]
        det_times = Float64[]
        det_achieved_ranks = Vector{Int}[]
        det_memory_usage = Int[]
        
        for trial in 1:n_trials
            try
                # Compute Hadamard product
                product = ttvectors[1] * ttvectors[2] * ttvectors[3]
                
                # Deterministic rounding with timing
                timer = TimerOutput()
                @timeit timer "tt_rounding" begin
                    compressed = tt_rounding(product; rmax=target_rank)
                end
                
                # Measure memory usage separately
                allocs = @allocated begin
                    product_copy = ttvectors[1] * ttvectors[2] * ttvectors[3]
                    tt_rounding(product_copy; rmax=target_rank)
                end
                
                # Compute error against exact solution
                error = norm(exact_solution - compressed) / exact_norm
                
                # Extract timing from TimerOutput
                main_time = TimerOutputs.time(timer["tt_rounding"]) / 1e9  # Convert to seconds
                
                # Store results
                push!(det_errors, error)
                push!(det_times, main_time)
                push!(det_achieved_ranks, compressed.ttv_rks)
                push!(det_memory_usage, allocs)
                
                println("  Trial $trial:")
                println("    Achieved ranks: $(compressed.ttv_rks)")
                println("    Relative error: $(@sprintf "%.4e" error)")
                println("    Total time: $(@sprintf "%.3f" main_time*1000) ms")
                println("    Memory: $(@sprintf "%.2f" allocs/1024^2) MB")
                
            catch e
                println("  Trial $trial: ✗ Failed with error: $e")
            end
        end
        
        if !isempty(det_errors)
            # Store deterministic results
            rank_results["deterministic"] = Dict(
                "method" => "deterministic",
                "method_name" => "tt_rounding",
                "errors" => det_errors,
                "main_times" => det_times,
                "memory_usage" => det_memory_usage,
                "achieved_ranks" => det_achieved_ranks,
                "median_error" => median(det_errors),
                "std_error" => std(det_errors),
                "median_time" => median(det_times),
                "std_time" => std(det_times),
                "median_memory" => median(det_memory_usage),
                "success_rate" => length(det_errors) / n_trials
            )
            
            # Summary for deterministic method
            println("  Summary:")
            println("    median error: $(@sprintf "%.4e" median(det_errors)) ± $(@sprintf "%.3e" std(det_errors))")
            println("    median total time: $(@sprintf "%.3f" median(det_times)*1000) ± $(@sprintf "%.3f" std(det_times)*1000) ms")
            println("    median memory: $(@sprintf "%.2f" median(det_memory_usage)/1024^2) MB")
            println("    Success rate: $(length(det_errors))/$n_trials")
        else
            println("  ✗ All trials failed for deterministic method")
            rank_results["deterministic"] = Dict(
                "method" => "deterministic",
                "method_name" => "tt_rounding",
                "success_rate" => 0.0
            )
        end
        println()
        
        for block_rks in block_rks_options
            block_name = if block_rks == 1
                "Khatri-Rao (block_rks=1)"
            elseif block_rks == 4
                "Small blocks (block_rks=4)" 
            elseif block_rks == 8
                "Large blocks (block_rks=8)"
            else
                "Custom (block_rks=$block_rks)"
            end
            
            println("\n--- $block_name ---")
            
            # Test across multiple trials for stability
            errors = Float64[]
            timer_results = TimerOutput[]
            achieved_ranks = Vector{Int}[]
            memory_usage = Int[]
            

            ttrand_rounding(ttvectors, target_rks; block_rks=block_rks, seed=seed)


            for trial in 1:n_trials
                try
                    # Randomized sketching with detailed timing
                    timer = TimerOutput()
                    
                    compressed = ttrand_rounding(
                        ttvectors, 
                        target_rks; 
                        block_rks=block_rks, 
                        seed=seed + trial*100, 
                        timer=timer
                    )
                    
                    # Measure memory usage separately (without timer overhead)
                    allocs = @allocated ttrand_rounding(
                        ttvectors, 
                        target_rks; 
                        block_rks=block_rks, 
                        seed=seed + trial*100
                    )
                    
                    # Compute error against exact solution
                    error = norm(exact_solution - compressed) / exact_norm
                    
                    # Store results
                    push!(errors, error)
                    push!(timer_results, timer)
                    push!(achieved_ranks, compressed.ttv_rks)
                    push!(memory_usage, allocs)
                    
                    # Extract main timing from TimerOutput
                    main_time = TimerOutputs.time(timer["ttrand_rounding"]) / 1e9  # Convert to seconds
                    
                    println("  Trial $trial:")
                    println("    Achieved ranks: $(compressed.ttv_rks)")
                    println("    Relative error: $(@sprintf "%.4e" error)")
                    println("    Total time: $(@sprintf "%.3f" main_time*1000) ms")
                    println("    Memory: $(@sprintf "%.2f" allocs/1024^2) MB")
                    
                    # Show detailed timing breakdown for first trial
                    if trial == 1
                        println("    Timing breakdown:")
                        for (name, nested_timer) in timer.inner_timers["ttrand_rounding"].inner_timers
                            nested_time = TimerOutputs.time(nested_timer) / 1e9
                            println("      $name: $(@sprintf "%.3f" nested_time*1000) ms")
                        end
                    end
                    
                catch e
                    println("  Trial $trial: ✗ Failed with error: $e")
                end
            end
            
            if !isempty(errors)
                # Extract timing data from TimerOutput results
                main_times = [TimerOutputs.time(timer["ttrand_rounding"]) / 1e9 for timer in timer_results]
                sketch_times = Float64[]
                orthog_times = Float64[]
                
                # Extract detailed timing components
                for timer in timer_results
                    ttrand_timer = timer["ttrand_rounding"]
                    if haskey(ttrand_timer.inner_timers, "reverse_sketch")
                        push!(sketch_times, TimerOutputs.time(ttrand_timer["reverse_sketch"]) / 1e9)
                    end
                    if haskey(ttrand_timer.inner_timers, "orthogonalization sweep")
                        push!(orthog_times, TimerOutputs.time(ttrand_timer["orthogonalization sweep"]) / 1e9)
                    elseif haskey(ttrand_timer.inner_timers, "orthogonalization")
                        push!(orthog_times, TimerOutputs.time(ttrand_timer["orthogonalization"]) / 1e9)
                    end
                end
                
                # Store results for this block_rks configuration
                rank_results["block_rks_$block_rks"] = Dict(
                    "block_rks" => block_rks,
                    "block_name" => block_name,
                    "errors" => errors,
                    "main_times" => main_times,
                    "sketch_times" => sketch_times,
                    "orthog_times" => orthog_times,
                    "memory_usage" => memory_usage,
                    "achieved_ranks" => achieved_ranks,
                    "timer_results" => timer_results,
                    "median_error" => median(errors),
                    "std_error" => std(errors),
                    "median_time" => median(main_times),
                    "std_time" => std(main_times),
                    "median_sketch_time" => isempty(sketch_times) ? 0.0 : median(sketch_times),
                    "median_orthog_time" => isempty(orthog_times) ? 0.0 : median(orthog_times),
                    "median_memory" => median(memory_usage),
                    "success_rate" => length(errors) / n_trials
                )
                
                # Summary for this configuration
                println("  Summary:")
                println("    median error: $(@sprintf "%.4e" median(errors)) ± $(@sprintf "%.3e" std(errors))")
                println("    median total time: $(@sprintf "%.3f" median(main_times)*1000) ± $(@sprintf "%.3f" std(main_times)*1000) ms")
                if !isempty(sketch_times)
                    println("    median sketch time: $(@sprintf "%.3f" median(sketch_times)*1000) ms")
                end
                if !isempty(orthog_times)
                    println("    median orthog time: $(@sprintf "%.3f" median(orthog_times)*1000) ms")
                end
                println("    median memory: $(@sprintf "%.2f" median(memory_usage)/1024^2) MB")
                println("    Success rate: $(length(errors))/$n_trials")
            else
                println("  ✗ All trials failed for $block_name")
                rank_results["block_rks_$block_rks"] = Dict(
                    "block_rks" => block_rks,
                    "block_name" => block_name,
                    "success_rate" => 0.0
                )
            end
        end
        
        results["target_rank_$target_rank"] = rank_results
    end
    
    return results
end

"""
Analyze and summarize benchmark results
"""
function analyze_sketching_results(results::Dict{String, Any})
    println("\n" * "="^80)
    println("HADAMARD PRODUCT SKETCHING ANALYSIS")
    println("="^80)
    
    # Summary table
    println("\nPerformance Summary:")
    println("Target Rank | Method            | median Error  | Total Time (ms) | Sketch Time | Orthog Time | Memory (MB) | Success")
    println("----------- | ----------------- | ----------- | --------------- | ----------- | ----------- | ----------- | -------")
    
    for (rank_key, rank_data) in results
        if startswith(rank_key, "target_rank_")
            target_rank = rank_data["target_rank"]
            
            # Show deterministic results first
            if haskey(rank_data, "deterministic")
                data = rank_data["deterministic"]
                if data["success_rate"] > 0
                    @printf("%-11d | %-17s | %10.3e | %14.2f | %10s | %10s | %10.2f | %d/%d\n",
                        target_rank, 
                        "Deterministic",
                        data["median_error"],
                        data["median_time"]*1000,
                        "N/A",
                        "N/A",
                        data["median_memory"]/1024^2,
                        round(Int, data["success_rate"] * length(get(data, "errors", []))),
                        length(get(data, "errors", []))
                    )
                else
                    @printf("%-11d | %-17s | %10s | %14s | %10s | %10s | %10s | 0/3\n",
                        target_rank, "Deterministic", "FAILED", "FAILED", "FAILED", "FAILED", "FAILED")
                end
            end
            
            # Extract block_rks results in order
            block_configs = [(1, "Khatri-Rao"), (4, "Small blocks"), (8, "Large blocks")]
            
            for (block_rks, config_name) in block_configs
                block_key = "block_rks_$block_rks"
                if haskey(rank_data, block_key)
                    data = rank_data[block_key]
                    if data["success_rate"] > 0
                        sketch_time_str = if haskey(data, "median_sketch_time") && data["median_sketch_time"] > 0
                            @sprintf("%.2f ms", data["median_sketch_time"]*1000)
                        else
                            "N/A"
                        end
                        orthog_time_str = if haskey(data, "median_orthog_time") && data["median_orthog_time"] > 0
                            @sprintf("%.2f ms", data["median_orthog_time"]*1000)
                        else
                            "N/A"
                        end
                        
                        @printf("%-11d | %-17s | %10.3e | %14.2f | %10s | %10s | %10.2f | %d/%d\n",
                            target_rank, 
                            config_name,
                            data["median_error"],
                            data["median_time"]*1000,
                            sketch_time_str,
                            orthog_time_str,
                            data["median_memory"]/1024^2,
                            round(Int, data["success_rate"] * length(get(data, "errors", []))),
                            length(get(data, "errors", []))
                        )
                    else
                        @printf("%-11d | %-17s | %10s | %14s | %10s | %10s | %10s | 0/3\n",
                            target_rank, config_name, "FAILED", "FAILED", "FAILED", "FAILED", "FAILED")
                    end
                end
            end
            println()
        end
    end
    
    # Best performance analysis
    println("Best Performance by Metric:")
    
    best_accuracy = (config="", target_rank=0, block_rks=0, error=Inf)
    best_speed = (config="", target_rank=0, block_rks=0, time=Inf)
    best_memory = (config="", target_rank=0, block_rks=0, memory=Inf)
    
    for (rank_key, rank_data) in results
        if startswith(rank_key, "target_rank_")
            target_rank = rank_data["target_rank"]
            
            # Check deterministic algorithm
            if haskey(rank_data, "deterministic") && rank_data["deterministic"]["success_rate"] > 0
                det_data = rank_data["deterministic"]
                config_name = "Deterministic"
                
                # Check accuracy
                if det_data["median_error"] < best_accuracy.error
                    best_accuracy = (config=config_name, target_rank=target_rank, 
                                   block_rks=0, error=det_data["median_error"])
                end
                
                # Check speed  
                if det_data["median_time"] < best_speed.time
                    best_speed = (config=config_name, target_rank=target_rank,
                                block_rks=0, time=det_data["median_time"])
                end
                
                # Check memory
                if det_data["median_memory"] < best_memory.memory
                    best_memory = (config=config_name, target_rank=target_rank,
                                 block_rks=0, memory=det_data["median_memory"])
                end
            end
            
            for (block_key, block_data) in rank_data
                if startswith(block_key, "block_rks_") && block_data["success_rate"] > 0
                    block_rks = block_data["block_rks"]
                    config_name = block_data["block_name"]
                    
                    # Check accuracy
                    if block_data["median_error"] < best_accuracy.error
                        best_accuracy = (config=config_name, target_rank=target_rank, 
                                       block_rks=block_rks, error=block_data["median_error"])
                    end
                    
                    # Check speed  
                    if block_data["median_time"] < best_speed.time
                        best_speed = (config=config_name, target_rank=target_rank,
                                    block_rks=block_rks, time=block_data["median_time"])
                    end
                    
                    # Check memory
                    if block_data["median_memory"] < best_memory.memory
                        best_memory = (config=config_name, target_rank=target_rank,
                                     block_rks=block_rks, memory=block_data["median_memory"])
                    end
                end
            end
        end
    end
    
    println("  Best accuracy: $(best_accuracy.config) (target_rank=$(best_accuracy.target_rank), error=$(@sprintf "%.2e" best_accuracy.error))")
    println("  Best speed: $(best_speed.config) (target_rank=$(best_speed.target_rank), time=$(@sprintf "%.2f" best_speed.time*1000)ms)")
    println("  Best memory: $(best_memory.config) (target_rank=$(best_memory.target_rank), mem=$(@sprintf "%.1f" best_memory.memory/1024^2)MB)")
    
    # Recommendations
    println("\nRecommendations:")
    println("  • For highest accuracy: Use large blocks (block_rks=8) with sufficient target rank")
    println("  • For balanced performance: Use small blocks (block_rks=4) as good compromise")
    println("  • For memory efficiency: Use Khatri-Rao structure (block_rks=1)")
    println("  • Block structure has most impact at moderate target ranks (8-32)")
    
    return results
end

"""
Create plots comparing different block rank configurations
"""
function create_hadamard_benchmark_plots(results::Dict{String, Any}, save_dir="out/hadamard_benchmarks")
    println("Creating benchmark comparison plots...")
    
    mkpath(save_dir)
    
    # Extract data for plotting
    target_ranks = Int[]
    
    # Data arrays for each configuration
    deterministic_errors = Float64[]
    khatri_rao_errors = Float64[]
    small_blocks_errors = Float64[]
    large_blocks_errors = Float64[]
    
    deterministic_times = Float64[]
    khatri_rao_times = Float64[]
    small_blocks_times = Float64[]
    large_blocks_times = Float64[]
    
    # Error bar data (for 20th and 80th percentiles)
    deterministic_error_bars = Tuple{Float64, Float64}[]
    khatri_rao_error_bars = Tuple{Float64, Float64}[]
    small_blocks_error_bars = Tuple{Float64, Float64}[]
    large_blocks_error_bars = Tuple{Float64, Float64}[]
    
    deterministic_time_bars = Tuple{Float64, Float64}[]
    khatri_rao_time_bars = Tuple{Float64, Float64}[]
    small_blocks_time_bars = Tuple{Float64, Float64}[]
    large_blocks_time_bars = Tuple{Float64, Float64}[]
    
    # Extract data from results
    for (rank_key, rank_data) in results
        if startswith(rank_key, "target_rank_")
            target_rank = rank_data["target_rank"]
            push!(target_ranks, target_rank)
            
            # Extract deterministic data
            if haskey(rank_data, "deterministic") && rank_data["deterministic"]["success_rate"] > 0
                det_data = rank_data["deterministic"]
                push!(deterministic_errors, det_data["median_error"])
                push!(deterministic_times, det_data["median_time"] * 1000)  # Convert to ms
                
                # Calculate error bars (20th and 80th percentiles)
                if haskey(det_data, "errors") && length(det_data["errors"]) > 1
                    errors_sorted = sort(det_data["errors"])
                    times_sorted = sort(det_data["main_times"]) * 1000  # Convert to ms
                    
                    error_20 = quantile(errors_sorted, 0.2)
                    error_80 = quantile(errors_sorted, 0.8)
                    time_20 = quantile(times_sorted, 0.2)
                    time_80 = quantile(times_sorted, 0.8)
                    
                    push!(deterministic_error_bars, (error_20, error_80))
                    push!(deterministic_time_bars, (time_20, time_80))
                else
                    push!(deterministic_error_bars, (det_data["median_error"], det_data["median_error"]))
                    push!(deterministic_time_bars, (det_data["median_time"] * 1000, det_data["median_time"] * 1000))
                end
            else
                push!(deterministic_errors, NaN)
                push!(deterministic_times, NaN)
                push!(deterministic_error_bars, (NaN, NaN))
                push!(deterministic_time_bars, (NaN, NaN))
            end
            
            # Extract error and timing data for each block configuration
            for (block_rks, data_array, time_array, error_bars_array, time_bars_array) in [
                (1, khatri_rao_errors, khatri_rao_times, khatri_rao_error_bars, khatri_rao_time_bars),
                (4, small_blocks_errors, small_blocks_times, small_blocks_error_bars, small_blocks_time_bars), 
                (8, large_blocks_errors, large_blocks_times, large_blocks_error_bars, large_blocks_time_bars)
            ]
                block_key = "block_rks_$block_rks"
                if haskey(rank_data, block_key) && rank_data[block_key]["success_rate"] > 0
                    block_data = rank_data[block_key]
                    push!(data_array, block_data["median_error"])
                    push!(time_array, block_data["median_time"] * 1000)  # Convert to ms
                    
                    # Calculate error bars (20th and 80th percentiles)
                    if haskey(block_data, "errors") && length(block_data["errors"]) > 1
                        errors_sorted = sort(block_data["errors"])
                        times_sorted = sort(block_data["main_times"]) * 1000  # Convert to ms
                        
                        error_20 = quantile(errors_sorted, 0.2)
                        error_80 = quantile(errors_sorted, 0.8)
                        time_20 = quantile(times_sorted, 0.2)
                        time_80 = quantile(times_sorted, 0.8)
                        
                        push!(error_bars_array, (error_20, error_80))
                        push!(time_bars_array, (time_20, time_80))
                    else
                        push!(error_bars_array, (block_data["median_error"], block_data["median_error"]))
                        push!(time_bars_array, (block_data["median_time"] * 1000, block_data["median_time"] * 1000))
                    end
                else
                    push!(data_array, NaN)
                    push!(time_array, NaN)
                    push!(error_bars_array, (NaN, NaN))
                    push!(time_bars_array, (NaN, NaN))
                end
            end
        end
    end
    
    # Sort by target rank
    sort_idx = sortperm(target_ranks)
    target_ranks = target_ranks[sort_idx]
    deterministic_errors = deterministic_errors[sort_idx]
    khatri_rao_errors = khatri_rao_errors[sort_idx]
    small_blocks_errors = small_blocks_errors[sort_idx]
    large_blocks_errors = large_blocks_errors[sort_idx]
    deterministic_times = deterministic_times[sort_idx]
    khatri_rao_times = khatri_rao_times[sort_idx]
    small_blocks_times = small_blocks_times[sort_idx]
    large_blocks_times = large_blocks_times[sort_idx]
    
    # Sort error bar arrays
    deterministic_error_bars = deterministic_error_bars[sort_idx]
    khatri_rao_error_bars = khatri_rao_error_bars[sort_idx]
    small_blocks_error_bars = small_blocks_error_bars[sort_idx]
    large_blocks_error_bars = large_blocks_error_bars[sort_idx]
    deterministic_time_bars = deterministic_time_bars[sort_idx]
    khatri_rao_time_bars = khatri_rao_time_bars[sort_idx]
    small_blocks_time_bars = small_blocks_time_bars[sort_idx]
    large_blocks_time_bars = large_blocks_time_bars[sort_idx]
    
    # Helper function to add error bars
    function add_error_bars!(plot_obj, x_vals, y_vals, error_bars, valid_mask, color, alpha=0.3)
        if any(valid_mask)
            x_valid = x_vals[valid_mask]
            y_valid = y_vals[valid_mask]
            error_bars_valid = error_bars[valid_mask]
            
            # Extract lower and upper bounds
            lower_bounds = [eb[1] for eb in error_bars_valid]
            upper_bounds = [eb[2] for eb in error_bars_valid]
            
            # Add error bars as ribbons
            plot!(plot_obj, x_valid, y_valid, 
                 ribbon=(y_valid .- lower_bounds, upper_bounds .- y_valid),
                 fillalpha=alpha, color=color, linewidth=0, label="")
        end
    end
    
    # Accuracy comparison plot
    println("Creating accuracy comparison plot...")
    p1 = plot(title="QuanticsTCI Hadamard Product: Accuracy vs Target Rank", 
             xlabel="Target Rank", ylabel="Relative Error", 
             yscale=:log10, legend=:topright, 
             grid=true, gridwidth=1, gridcolor=:lightgray,
             size=(800, 600))
    
    # Filter out NaN values for plotting
    valid_det = .!isnan.(deterministic_errors)
    valid_kr = .!isnan.(khatri_rao_errors)
    valid_sb = .!isnan.(small_blocks_errors)
    valid_lb = .!isnan.(large_blocks_errors)
    
    # Add error bars first (so they appear behind the lines)
    add_error_bars!(p1, target_ranks, deterministic_errors, deterministic_error_bars, valid_det, :black, 0.2)
    add_error_bars!(p1, target_ranks, khatri_rao_errors, khatri_rao_error_bars, valid_kr, :red, 0.2)
    add_error_bars!(p1, target_ranks, small_blocks_errors, small_blocks_error_bars, valid_sb, :blue, 0.2)
    add_error_bars!(p1, target_ranks, large_blocks_errors, large_blocks_error_bars, valid_lb, :green, 0.2)
    
    # Add main lines with markers
    if any(valid_det)
        plot!(p1, target_ranks[valid_det], deterministic_errors[valid_det], 
             label="Deterministic (tt_rounding)", marker=:star, linewidth=3, 
             color=:black, markersize=8)
    end
    
    if any(valid_kr)
        plot!(p1, target_ranks[valid_kr], khatri_rao_errors[valid_kr], 
             label="Khatri-Rao (block_rks=1)", marker=:circle, linewidth=2, 
             color=:red, markersize=6)
    end
    
    if any(valid_sb)
        plot!(p1, target_ranks[valid_sb], small_blocks_errors[valid_sb], 
             label="Small blocks (block_rks=4)", marker=:square, linewidth=2, 
             color=:blue, markersize=6)
    end
    
    if any(valid_lb)
        plot!(p1, target_ranks[valid_lb], large_blocks_errors[valid_lb], 
             label="Large blocks (block_rks=8)", marker=:diamond, linewidth=2, 
             color=:green, markersize=6)
    end
    
    accuracy_file = joinpath(save_dir, "hadamard_accuracy_comparison.png")
    savefig(p1, accuracy_file)
    println("Accuracy plot saved: $accuracy_file")
    
    # Timing comparison plot
    println("Creating timing comparison plot...")
    p2 = plot(title="QuanticsTCI Hadamard Product: Timing vs Target Rank", 
             xlabel="Target Rank", ylabel="Time (milliseconds)", 
             legend=:topleft,
             grid=true, gridwidth=1, gridcolor=:lightgray,
             size=(800, 600))
    
    # Filter out NaN values for timing plot
    valid_det_time = .!isnan.(deterministic_times)
    valid_kr_time = .!isnan.(khatri_rao_times)
    valid_sb_time = .!isnan.(small_blocks_times)
    valid_lb_time = .!isnan.(large_blocks_times)
    
    # Add timing error bars first (so they appear behind the lines)
    add_error_bars!(p2, target_ranks, deterministic_times, deterministic_time_bars, valid_det_time, :black, 0.2)
    add_error_bars!(p2, target_ranks, khatri_rao_times, khatri_rao_time_bars, valid_kr_time, :red, 0.2)
    add_error_bars!(p2, target_ranks, small_blocks_times, small_blocks_time_bars, valid_sb_time, :blue, 0.2)
    add_error_bars!(p2, target_ranks, large_blocks_times, large_blocks_time_bars, valid_lb_time, :green, 0.2)
    
    if any(valid_det_time)
        plot!(p2, target_ranks[valid_det_time], deterministic_times[valid_det_time], 
             label="Deterministic (tt_rounding)", marker=:star, linewidth=3, 
             color=:black, markersize=8)
    end
    
    if any(valid_kr_time)
        plot!(p2, target_ranks[valid_kr_time], khatri_rao_times[valid_kr_time], 
             label="Khatri-Rao (block_rks=1)", marker=:circle, linewidth=2, 
             color=:red, markersize=6)
    end
    
    if any(valid_sb_time)
        plot!(p2, target_ranks[valid_sb_time], small_blocks_times[valid_sb_time], 
             label="Small blocks (block_rks=4)", marker=:square, linewidth=2, 
             color=:blue, markersize=6)
    end
    
    if any(valid_lb_time)
        plot!(p2, target_ranks[valid_lb_time], large_blocks_times[valid_lb_time], 
             label="Large blocks (block_rks=8)", marker=:diamond, linewidth=2, 
             color=:green, markersize=6)
    end
    
    timing_file = joinpath(save_dir, "hadamard_timing_comparison.png")
    savefig(p2, timing_file)
    println("Timing plot saved: $timing_file")
    
    # Combined plot with subplots
    println("Creating combined comparison plot...")
    p_combined = plot(p1, p2, layout=(1,2), size=(1600, 600),
                     plot_title="QuanticsTCI Hadamard Product Sketching Benchmark")
    
    combined_file = joinpath(save_dir, "hadamard_benchmark_comparison.png")
    savefig(p_combined, combined_file)
    println("Combined plot saved: $combined_file")
    
    # Create timing breakdown plot
    println("Creating timing breakdown plot...")
    
    # Extract sketch and orthogonalization times with error bars
    sketch_times_kr = Float64[]
    sketch_times_sb = Float64[]
    sketch_times_lb = Float64[]
    orthog_times_kr = Float64[]
    orthog_times_sb = Float64[]
    orthog_times_lb = Float64[]
    
    # Error bar data for sketch and orthog times
    sketch_error_bars_kr = Tuple{Float64, Float64}[]
    sketch_error_bars_sb = Tuple{Float64, Float64}[]
    sketch_error_bars_lb = Tuple{Float64, Float64}[]
    orthog_error_bars_kr = Tuple{Float64, Float64}[]
    orthog_error_bars_sb = Tuple{Float64, Float64}[]
    orthog_error_bars_lb = Tuple{Float64, Float64}[]
    
    for (rank_key, rank_data) in results
        if startswith(rank_key, "target_rank_")
            for (block_rks, sketch_array, orthog_array, sketch_bars_array, orthog_bars_array) in [
                (1, sketch_times_kr, orthog_times_kr, sketch_error_bars_kr, orthog_error_bars_kr),
                (4, sketch_times_sb, orthog_times_sb, sketch_error_bars_sb, orthog_error_bars_sb),
                (8, sketch_times_lb, orthog_times_lb, sketch_error_bars_lb, orthog_error_bars_lb)
            ]
                block_key = "block_rks_$block_rks"
                if haskey(rank_data, block_key) && rank_data[block_key]["success_rate"] > 0
                    data = rank_data[block_key]
                    push!(sketch_array, get(data, "median_sketch_time", 0.0) * 1000)  # Convert to ms
                    push!(orthog_array, get(data, "median_orthog_time", 0.0) * 1000)  # Convert to ms
                    
                    # Calculate error bars for sketch and orthog times
                    if haskey(data, "sketch_times") && length(data["sketch_times"]) > 1
                        sketch_times_sorted = sort(data["sketch_times"]) * 1000  # Convert to ms
                        sketch_20 = quantile(sketch_times_sorted, 0.2)
                        sketch_80 = quantile(sketch_times_sorted, 0.8)
                        push!(sketch_bars_array, (sketch_20, sketch_80))
                    else
                        median_sketch = get(data, "median_sketch_time", 0.0) * 1000
                        push!(sketch_bars_array, (median_sketch, median_sketch))
                    end
                    
                    if haskey(data, "orthog_times") && length(data["orthog_times"]) > 1
                        orthog_times_sorted = sort(data["orthog_times"]) * 1000  # Convert to ms
                        orthog_20 = quantile(orthog_times_sorted, 0.2)
                        orthog_80 = quantile(orthog_times_sorted, 0.8)
                        push!(orthog_bars_array, (orthog_20, orthog_80))
                    else
                        median_orthog = get(data, "median_orthog_time", 0.0) * 1000
                        push!(orthog_bars_array, (median_orthog, median_orthog))
                    end
                else
                    push!(sketch_array, NaN)
                    push!(orthog_array, NaN)
                    push!(sketch_bars_array, (NaN, NaN))
                    push!(orthog_bars_array, (NaN, NaN))
                end
            end
        end
    end
    
    # Sort timing breakdown data
    sketch_times_kr = sketch_times_kr[sort_idx]
    sketch_times_sb = sketch_times_sb[sort_idx]  
    sketch_times_lb = sketch_times_lb[sort_idx]
    orthog_times_kr = orthog_times_kr[sort_idx]
    orthog_times_sb = orthog_times_sb[sort_idx]
    orthog_times_lb = orthog_times_lb[sort_idx]
    
    # Sort error bar arrays for timing breakdown
    sketch_error_bars_kr = sketch_error_bars_kr[sort_idx]
    sketch_error_bars_sb = sketch_error_bars_sb[sort_idx]
    sketch_error_bars_lb = sketch_error_bars_lb[sort_idx]
    orthog_error_bars_kr = orthog_error_bars_kr[sort_idx]
    orthog_error_bars_sb = orthog_error_bars_sb[sort_idx]
    orthog_error_bars_lb = orthog_error_bars_lb[sort_idx]
    
    p3 = plot(title="Timing Breakdown: Sketch vs Orthogonalization", 
             xlabel="Target Rank", ylabel="Time (milliseconds)",
             legend=:topleft,
             grid=true, gridwidth=1, gridcolor=:lightgray,
             size=(800, 600))
    
    # Filter out NaN values for timing breakdown
    valid_sketch_kr = .!isnan.(sketch_times_kr)
    valid_sketch_sb = .!isnan.(sketch_times_sb)
    valid_sketch_lb = .!isnan.(sketch_times_lb)
    valid_orthog_kr = .!isnan.(orthog_times_kr)
    valid_orthog_sb = .!isnan.(orthog_times_sb)
    valid_orthog_lb = .!isnan.(orthog_times_lb)
    
    # Add error bars first (so they appear behind the lines)
    add_error_bars!(p3, target_ranks, sketch_times_kr, sketch_error_bars_kr, valid_sketch_kr, :red, 0.15)
    add_error_bars!(p3, target_ranks, sketch_times_sb, sketch_error_bars_sb, valid_sketch_sb, :blue, 0.15)
    add_error_bars!(p3, target_ranks, sketch_times_lb, sketch_error_bars_lb, valid_sketch_lb, :green, 0.15)
    add_error_bars!(p3, target_ranks, orthog_times_kr, orthog_error_bars_kr, valid_orthog_kr, :red, 0.15)
    add_error_bars!(p3, target_ranks, orthog_times_sb, orthog_error_bars_sb, valid_orthog_sb, :blue, 0.15)
    add_error_bars!(p3, target_ranks, orthog_times_lb, orthog_error_bars_lb, valid_orthog_lb, :green, 0.15)
    
    # Plot sketch times
    if any(valid_sketch_kr)
        plot!(p3, target_ranks, sketch_times_kr, label="Khatri-Rao (sketch)", 
             color=:red, linestyle=:dash, linewidth=2, marker=:circle)
    end
    if any(valid_sketch_sb)
        plot!(p3, target_ranks, sketch_times_sb, label="Small blocks (sketch)", 
             color=:blue, linestyle=:dash, linewidth=2, marker=:square)
    end
    if any(valid_sketch_lb)
        plot!(p3, target_ranks, sketch_times_lb, label="Large blocks (sketch)", 
             color=:green, linestyle=:dash, linewidth=2, marker=:diamond)
    end
    
    # Plot orthogonalization times
    if any(valid_orthog_kr)
        plot!(p3, target_ranks, orthog_times_kr, label="Khatri-Rao (QR)", 
             color=:red, linestyle=:solid, linewidth=2, marker=:circle)
    end
    if any(valid_orthog_sb)
        plot!(p3, target_ranks, orthog_times_sb, label="Small blocks (QR)", 
             color=:blue, linestyle=:solid, linewidth=2, marker=:square)
    end
    if any(valid_orthog_lb)
        plot!(p3, target_ranks, orthog_times_lb, label="Large blocks (QR)", 
             color=:green, linestyle=:solid, linewidth=2, marker=:diamond)
    end
    
    breakdown_file = joinpath(save_dir, "hadamard_timing_breakdown.png")
    savefig(p3, breakdown_file)
    println("Timing breakdown plot saved: $breakdown_file")

    # Memory usage comparison plot
    println("Creating memory usage comparison plot...")

    # Extract memory usage data
    deterministic_memory = Float64[]
    khatri_rao_memory = Float64[]
    small_blocks_memory = Float64[]
    large_blocks_memory = Float64[]

    # Error bar data for memory (20th and 80th percentiles)
    deterministic_memory_bars = Tuple{Float64, Float64}[]
    khatri_rao_memory_bars = Tuple{Float64, Float64}[]
    small_blocks_memory_bars = Tuple{Float64, Float64}[]
    large_blocks_memory_bars = Tuple{Float64, Float64}[]

    for (rank_key, rank_data) in results
        if startswith(rank_key, "target_rank_")
            # Extract deterministic memory data
            if haskey(rank_data, "deterministic") && rank_data["deterministic"]["success_rate"] > 0
                det_data = rank_data["deterministic"]
                push!(deterministic_memory, det_data["median_memory"] / 1024^2)  # Convert to MB

                # Calculate error bars
                if haskey(det_data, "memory_usage") && length(det_data["memory_usage"]) > 1
                    memory_sorted = sort(det_data["memory_usage"]) / 1024^2  # Convert to MB
                    memory_20 = quantile(memory_sorted, 0.2)
                    memory_80 = quantile(memory_sorted, 0.8)
                    push!(deterministic_memory_bars, (memory_20, memory_80))
                else
                    median_mem = det_data["median_memory"] / 1024^2
                    push!(deterministic_memory_bars, (median_mem, median_mem))
                end
            else
                push!(deterministic_memory, NaN)
                push!(deterministic_memory_bars, (NaN, NaN))
            end

            # Extract memory data for each block configuration
            for (block_rks, memory_array, memory_bars_array) in [
                (1, khatri_rao_memory, khatri_rao_memory_bars),
                (4, small_blocks_memory, small_blocks_memory_bars),
                (8, large_blocks_memory, large_blocks_memory_bars)
            ]
                block_key = "block_rks_$block_rks"
                if haskey(rank_data, block_key) && rank_data[block_key]["success_rate"] > 0
                    block_data = rank_data[block_key]
                    push!(memory_array, block_data["median_memory"] / 1024^2)  # Convert to MB

                    # Calculate error bars
                    if haskey(block_data, "memory_usage") && length(block_data["memory_usage"]) > 1
                        memory_sorted = sort(block_data["memory_usage"]) / 1024^2  # Convert to MB
                        memory_20 = quantile(memory_sorted, 0.2)
                        memory_80 = quantile(memory_sorted, 0.8)
                        push!(memory_bars_array, (memory_20, memory_80))
                    else
                        median_mem = block_data["median_memory"] / 1024^2
                        push!(memory_bars_array, (median_mem, median_mem))
                    end
                else
                    push!(memory_array, NaN)
                    push!(memory_bars_array, (NaN, NaN))
                end
            end
        end
    end

    # Sort memory data
    deterministic_memory = deterministic_memory[sort_idx]
    khatri_rao_memory = khatri_rao_memory[sort_idx]
    small_blocks_memory = small_blocks_memory[sort_idx]
    large_blocks_memory = large_blocks_memory[sort_idx]
    deterministic_memory_bars = deterministic_memory_bars[sort_idx]
    khatri_rao_memory_bars = khatri_rao_memory_bars[sort_idx]
    small_blocks_memory_bars = small_blocks_memory_bars[sort_idx]
    large_blocks_memory_bars = large_blocks_memory_bars[sort_idx]

    p4 = plot(title="QuanticsTCI Hadamard Product: Memory Usage vs Target Rank",
             xlabel="Target Rank", ylabel="Memory Usage (MB)",
             legend=:topleft,
             grid=true, gridwidth=1, gridcolor=:lightgray,
             size=(800, 600))

    # Filter out NaN values for memory plot
    valid_det_memory = .!isnan.(deterministic_memory)
    valid_kr_memory = .!isnan.(khatri_rao_memory)
    valid_sb_memory = .!isnan.(small_blocks_memory)
    valid_lb_memory = .!isnan.(large_blocks_memory)

    # Add memory error bars first
    add_error_bars!(p4, target_ranks, deterministic_memory, deterministic_memory_bars, valid_det_memory, :black, 0.2)
    add_error_bars!(p4, target_ranks, khatri_rao_memory, khatri_rao_memory_bars, valid_kr_memory, :red, 0.2)
    add_error_bars!(p4, target_ranks, small_blocks_memory, small_blocks_memory_bars, valid_sb_memory, :blue, 0.2)
    add_error_bars!(p4, target_ranks, large_blocks_memory, large_blocks_memory_bars, valid_lb_memory, :green, 0.2)

    if any(valid_det_memory)
        plot!(p4, target_ranks[valid_det_memory], deterministic_memory[valid_det_memory],
             label="Deterministic (tt_rounding)", marker=:star, linewidth=3,
             color=:black, markersize=8)
    end

    if any(valid_kr_memory)
        plot!(p4, target_ranks[valid_kr_memory], khatri_rao_memory[valid_kr_memory],
             label="Khatri-Rao (block_rks=1)", marker=:circle, linewidth=2,
             color=:red, markersize=6)
    end

    if any(valid_sb_memory)
        plot!(p4, target_ranks[valid_sb_memory], small_blocks_memory[valid_sb_memory],
             label="Small blocks (block_rks=4)", marker=:square, linewidth=2,
             color=:blue, markersize=6)
    end

    if any(valid_lb_memory)
        plot!(p4, target_ranks[valid_lb_memory], large_blocks_memory[valid_lb_memory],
             label="Large blocks (block_rks=8)", marker=:diamond, linewidth=2,
             color=:green, markersize=6)
    end

    memory_file = joinpath(save_dir, "hadamard_memory_comparison.png")
    savefig(p4, memory_file)
    println("Memory comparison plot saved: $memory_file")

    return accuracy_file, timing_file, combined_file, breakdown_file, memory_file
end

"""
Load benchmark results from JSON file if it exists
"""
function load_benchmark_results(save_dir="out/hadamard_benchmarks")
    results_file = joinpath(save_dir, "hadamard_benchmark_results.json")
    
    if isfile(results_file)
        println("Found existing results file: $results_file")
        try
            # Read and parse JSON file
            data = JSON3.read(read(results_file, String), allow_inf=true)
            println("Successfully loaded existing benchmark results")
            
            # Convert to regular Dict for better compatibility
            function convert_to_dict(obj)
                if obj isa JSON3.Object
                    return Dict{String, Any}(string(k) => convert_to_dict(v) for (k, v) in obj)
                elseif obj isa Vector
                    return [convert_to_dict(item) for item in obj]
                else
                    return obj
                end
            end
            
            results_dict = convert_to_dict(data[:results])
            metadata_dict = convert_to_dict(data[:metadata])
            
            return results_dict, metadata_dict
        catch e
            println("Warning: Could not load existing results file: $e")
            println("Will run new experiments instead")
            return nothing, nothing
        end
    else
        println("No existing results file found at: $results_file")
        return nothing, nothing
    end
end

"""
Save benchmark results to JSON file
"""
function save_benchmark_results(results::Dict{String, Any}, metadata::Dict{String, Any}, 
                               save_dir="out/hadamard_benchmarks")
    mkpath(save_dir)
    
    # Prepare data for JSON serialization (remove TimerOutput objects)
    json_results = Dict{String, Any}()
    
    for (key, value) in results
        if isa(value, Dict)
            json_results[key] = Dict{String, Any}()
            for (sub_key, sub_value) in value
                if sub_key == "timer_results"
                    # Skip TimerOutput objects
                    continue
                else
                    json_results[key][sub_key] = sub_value
                end
            end
        else
            json_results[key] = value
        end
    end
    
    # Add metadata
    full_results = Dict(
        "metadata" => metadata,
        "results" => json_results,
        "timestamp" => string(now())
    )
    
    results_file = joinpath(save_dir, "hadamard_benchmark_results.json")
    open(results_file, "w") do io
        JSON3.write(io, full_results, allow_inf=true)
    end
    
    println("Results saved to: $results_file")
    return results_file
end

"""
Main benchmark function
"""
function run_quantics_hadamard_benchmark(;
    R = 20,                              # Grid resolution: 2^R per dimension
    tolerance = 1e-10,                   # QuanticsTCI tolerance
    target_ranks = [8, 16, 32],          # Target compression ranks to test
    block_rks_options = [1, 4, 8],      # Block rank configurations
    n_trials = 3,                        # Number of trials per configuration
    seed = 1234,
    force_rerun = false                  # Force rerun experiments even if results exist
)
    Random.seed!(seed)
    
    println("QuanticsTCI Hadamard Product Sketching Benchmark")
    println("=" * "^" * "60")
    println("Grid resolution: R=$R ($(2^R)³ = $(2^(3*R)) points)")
    println("QuanticsTCI tolerance: $tolerance") 
    println("Target ranks: $target_ranks")
    println("Block rank options: $block_rks_options")
    println("Trials per config: $n_trials")
    println()
    
    # Check for existing results first (unless forced to rerun)
    results = nothing
    ttvectors = nothing
    exact_solution = nothing
    
    if !force_rerun
        println("Checking for existing benchmark results...")
        existing_results, existing_metadata = load_benchmark_results()
        
        # Verify that existing results match current parameters
        if existing_results !== nothing && existing_metadata !== nothing
            params_match = (
                existing_metadata["R"] == R &&
                existing_metadata["tolerance"] == tolerance &&
                existing_metadata["target_ranks"] == target_ranks &&
                existing_metadata["block_rks_options"] == block_rks_options &&
                existing_metadata["n_trials"] == n_trials &&
                existing_metadata["seed"] == seed
            )
            
            if params_match
                println("✓ Existing results match current parameters - using cached data")
                results = existing_results
                
                # We still need tensor trains for potential future use, but we can skip experiments
                println("Note: Tensor train data not loaded - only using benchmark results for plotting")
            else
                println("⚠ Existing results have different parameters - running new experiments")
                println("  Current: R=$R, tol=$tolerance, ranks=$target_ranks, blocks=$block_rks_options, trials=$n_trials, seed=$seed")
                println("  Existing: R=$(existing_metadata["R"]), tol=$(existing_metadata["tolerance"]), ranks=$(existing_metadata["target_ranks"]), blocks=$(existing_metadata["block_rks_options"]), trials=$(existing_metadata["n_trials"]), seed=$(existing_metadata["seed"])")
            end
        end
    else
        println("Force rerun enabled - skipping existing results check")
    end
    
    # Run experiments if no valid cached results
    if results === nothing
        println("\n=== Running New Experiments ===")
        
        # Step 1: Generate QuanticsTCI tensor trains
        ttvectors = generate_quantics_test_tensors(R; tolerance=tolerance, seed=seed)
        
        # Step 2: Compute exact reference solution
        exact_solution = compute_exact_hadamard(ttvectors; tol=1e-12)
        
        # Step 3: Benchmark structured sketching
        results = benchmark_structured_sketching(
            ttvectors, 
            exact_solution;
            target_ranks=target_ranks,
            block_rks_options=block_rks_options,
            n_trials=n_trials,
            seed=seed
        )
        
        # Step 4: Save new results
        metadata = Dict(
            "R" => R,
            "tolerance" => tolerance,
            "target_ranks" => target_ranks,
            "block_rks_options" => block_rks_options,
            "n_trials" => n_trials,
            "seed" => seed,
            "grid_size" => 2^(3*R),
            "description" => "QuanticsTCI Hadamard product sketching benchmark with block rank comparison"
        )
        
        results_file = save_benchmark_results(results, metadata)
        println("New benchmark data saved to: $results_file")
    end
    
    # Step 6: Analyze results
    final_results = analyze_sketching_results(results)
    
    # Step 7: Create plots
    println("\n" * "="^60)
    println("CREATING VISUALIZATION PLOTS")
    println("="^60)
    
    # Create visualization plots
    plot_files = create_hadamard_benchmark_plots(results)
    println("Plot files created: $(length(plot_files)) files")
    
    println("\n" * "="^80)
    println("QUANTICS HADAMARD SKETCHING BENCHMARK COMPLETE")
    println("="^80)
    println("Generated files:")
    for file in plot_files
        println("  • $file")
    end
    if results !== nothing
        println("  • out/hadamard_benchmarks/hadamard_benchmark_results.json")
    end
    
    return final_results, ttvectors, exact_solution
end

"""
Create plots from existing results without running experiments
"""
function create_plots_from_existing_results(save_dir="out/hadamard_benchmarks")
    println("Loading existing results and creating plots...")
    
    # Load existing results
    results, metadata = load_benchmark_results(save_dir)
    
    if results === nothing
        println("❌ No existing results found. Please run the benchmark first.")
        return nothing
    end
    
    println("✓ Successfully loaded existing results")
    println("Parameters: R=$(metadata["R"]), tolerance=$(metadata["tolerance"])")
    println("Target ranks: $(metadata["target_ranks"])")
    println("Block rank options: $(metadata["block_rks_options"])")
    println("Trials: $(metadata["n_trials"])")
    
    # Create plots
    println("\nCreating visualization plots...")
    plot_files = create_hadamard_benchmark_plots(results, save_dir)
    
    println("\n" * "="^60)
    println("PLOTS CREATED SUCCESSFULLY")
    println("="^60)
    println("Generated files:")
    for file in plot_files
        println("  • $file")
    end
    
    return plot_files
end

# Run benchmark when script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    results, ttvectors, exact_solution = run_quantics_hadamard_benchmark(
        R=20,  # Smaller scale for testing
        target_ranks=[8,16,24,32,40,48,56,64,72,80],
        block_rks_options=[1, 4, 8], 
        n_trials=100
    )
end