using Revise
using TensorTrains
using LinearAlgebra
using Random
using Printf
using QuanticsTCI
using Statistics
using TimerOutputs
using CairoMakie
using Colors
using JSON3
using Dates

# Configure CairoMakie for publication-quality plots
CairoMakie.activate!(type = "pdf")

# Define consistent styling theme
const PLOT_THEME = Theme(
    fontsize = 12,
    font = "Computer Modern",  # Good Computer Modern alternative
    linewidth = 2,
    markersize = 8,
    Axis = (
        titlesize = 12,
        xlabelsize = 11,
        ylabelsize = 11,
        xticklabelsize = 10,
        yticklabelsize = 10,
        spinewidth = 1.5,
        xtickwidth = 1.5,
        ytickwidth = 1.5,
        xgridwidth = 0.75,
        ygridwidth = 0.75,
        xgridcolor = :gray90,
        ygridcolor = :gray90,
        xgridvisible = true,
        ygridvisible = true,
        topspinevisible = true,
        rightspinevisible = true,
        leftspinecolor = :black,
        rightspinecolor = :black,
        topspinecolor = :black,
        bottomspinecolor = :black
    ),
    Legend = (
        labelsize = 11,
        titlesize = 12,
        framevisible = true,
        backgroundcolor = (:white, 0.9),
        framecolor = :black,
        framewidth = 1
    )
)
set_theme!(PLOT_THEME)

"""
QuanticsTCI Hadamard Product Sketch Benchmark

Tests structured sketching of Hadamard products using QuanticsTCI-derived tensor trains.
This follows the pattern from quantics_rounding_benchmark.jl but focuses on testing
the multi-TTvector Kronecker sketching approach for 3-term products.

The benchmark:
1. Uses QuanticsTCI to create tensor train representations of term1, term2, term3 functions
2. Tests ttrand_rounding((tt1, tt2, tt3), target_ranks; R=...) 
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
    results["target_ranks"] = target_ranks
    results["block_rks_options"] = block_rks_options
    
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
                println("  Warming up ttrand_rounding R=$block_rks...")
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
        
        # Test structured sketching with different block rank configurations
        for block_rks in block_rks_options
            block_name = "R=$block_rks"
            if block_rks == 1
                block_name = block_name*"(Khatri-Rao)"
            end
            
            println("\n--- $block_name ---")
            
            # Test across multiple trials for stability
            errors = Float64[]
            timer_results = TimerOutput[]
            achieved_ranks = Vector{Int}[]
            memory_usage = Int[]
            

            ttrand_rounding(ttvectors, target_rks; block_rks=block_rks, seed=seed) #Why is this here? It seems like a warm-up 

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
    
    best_accuracy = (config="", target_rank=0, R=0, error=Inf)
    best_speed = (config="", target_rank=0, R=0, time=Inf)
    best_memory = (config="", target_rank=0, R=0, memory=Inf)
    
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
                                   R=0, error=det_data["median_error"])
                end
                
                # Check speed  
                if det_data["median_time"] < best_speed.time
                    best_speed = (config=config_name, target_rank=target_rank,
                                R=0, time=det_data["median_time"])
                end
                
                # Check memory
                if det_data["median_memory"] < best_memory.memory
                    best_memory = (config=config_name, target_rank=target_rank,
                                 R=0, memory=det_data["median_memory"])
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
    println("  • For highest accuracy: Use large blocks (R=8) with sufficient target rank")
    println("  • For balanced performance: Use small blocks (R=4) as good compromise")
    println("  • For memory efficiency: Use Khatri-Rao structure (R=1)")
    println("  • Block structure has most impact at moderate target ranks (8-32)")
    
    return results
end

"""
Create plots comparing different block rank configurations using Makie
"""
function create_hadamard_benchmark_plots(results::Dict{String, Any}, dir="out/hadamard_benchmarks")
    println("Creating benchmark comparison plots...")
    save_dir = "$dir/plots"
    mkpath(save_dir)
    
    # Base palettes for dynamic generation
    base_colors = [:blue, :orange, :green, :red, :purple, :cyan, :brown, :magenta]
    base_markers = [:circle, :rect, :diamond, :utriangle, :dtriangle, :star5, :xcross, :hexagon]

    # Find all unique block ranks present in results
    all_block_rks = Int[]
    for (rank_key, rank_data) in results
        if startswith(rank_key, "target_rank_")
            for key in keys(rank_data)
                if startswith(key, "block_rks_")
                    push!(all_block_rks, rank_data[key]["block_rks"])
                end
            end
        end
    end
    unique!(all_block_rks)
    sort!(all_block_rks)

    # Dynamically build mapping dictionaries
    colors = Dict{Any, Symbol}()
    markers = Dict{Any, Symbol}()
    color_elements = MarkerElement[]
    color_labels = LaTeXString[]
    
    colors[:deterministic] = :black
    markers[:deterministic] = :star8
    push!(color_elements, MarkerElement(marker = :star8, color = :black, markersize = 10, strokecolor = :transparent))
    push!(color_labels, LaTeXString("Deterministic"))

    for (i, rk) in enumerate(all_block_rks)
        c = base_colors[mod1(i, length(base_colors))]
        m = base_markers[mod1(i, length(base_markers))]
        colors[rk] = c
        markers[rk] = m

        push!(color_elements, MarkerElement(marker = m, color = c, markersize = 10, strokecolor = :transparent))
        label_str = rk == 1 ? "R=1(Khatri-Rao)" : "R = $rk"
        push!(color_labels, LaTeXString(label_str))
    end
    
    # Extract data for plotting
    target_ranks = Int[]
    
    # Data arrays for each configuration
    deterministic_errors = Float64[]
    block_errors = Dict{Int, Vector{Float64}}()
    for rk in all_block_rks
        block_errors[rk] = Float64[]
    end
    
    deterministic_times = Float64[]
    block_times = Dict{Int, Vector{Float64}}()
    for rk in all_block_rks
        block_times[rk] = Float64[]
    end
    
    # Error bar data (for 25th and 75th percentiles)
    deterministic_error_bars = Tuple{Float64, Float64}[]
    block_error_bars = Dict{Int, Vector{Tuple{Float64, Float64}}}()
    for rk in all_block_rks
        block_error_bars[rk] = Tuple{Float64, Float64}[]
    end
    
    deterministic_time_bars = Tuple{Float64, Float64}[]
    block_time_bars = Dict{Int, Vector{Tuple{Float64, Float64}}}()
    for rk in all_block_rks
        block_time_bars[rk] = Tuple{Float64, Float64}[]
    end
    
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
                
                # Calculate error bars (25th and 75th percentiles)
                if haskey(det_data, "errors") && length(det_data["errors"]) > 1
                    errors_sorted = sort(det_data["errors"])
                    times_sorted = sort(det_data["main_times"]) * 1000  # Convert to ms
                    
                    error_25 = quantile(errors_sorted, 0.25)
                    error_75 = quantile(errors_sorted, 0.75)
                    time_25 = quantile(times_sorted, 0.25)
                    time_75 = quantile(times_sorted, 0.75)
                    
                    push!(deterministic_error_bars, (error_25, error_75))
                    push!(deterministic_time_bars, (time_25, time_75))
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
            for block_rks in all_block_rks
                block_key = "block_rks_$block_rks"
                if haskey(rank_data, block_key) && rank_data[block_key]["success_rate"] > 0
                    block_data = rank_data[block_key]
                    push!(block_errors[block_rks], block_data["median_error"])
                    push!(block_times[block_rks], block_data["median_time"] * 1000)  # Convert to ms
                    
                    # Calculate error bars (25th and 75th percentiles)
                    if haskey(block_data, "errors") && length(block_data["errors"]) > 1
                        errors_sorted = sort(block_data["errors"])
                        times_sorted = sort(block_data["main_times"]) * 1000  # Convert to ms
                        
                        error_25 = quantile(errors_sorted, 0.25)
                        error_75 = quantile(errors_sorted, 0.75)
                        time_25 = quantile(times_sorted, 0.25)
                        time_75 = quantile(times_sorted, 0.75)
                        
                        push!(block_error_bars[block_rks], (error_25, error_75))
                        push!(block_time_bars[block_rks], (time_25, time_75))
                    else
                        push!(block_error_bars[block_rks], (block_data["median_error"], block_data["median_error"]))
                        push!(block_time_bars[block_rks], (block_data["median_time"] * 1000, block_data["median_time"] * 1000))
                    end
                else
                    push!(block_errors[block_rks], NaN)
                    push!(block_times[block_rks], NaN)
                    push!(block_error_bars[block_rks], (NaN, NaN))
                    push!(block_time_bars[block_rks], (NaN, NaN))
                end
            end
        end
    end
    
    # Sort by target rank
    sort_idx = sortperm(target_ranks)
    target_ranks = target_ranks[sort_idx]
    deterministic_errors = deterministic_errors[sort_idx]
    deterministic_times = deterministic_times[sort_idx]
    deterministic_error_bars = deterministic_error_bars[sort_idx]
    deterministic_time_bars = deterministic_time_bars[sort_idx]

    for rk in all_block_rks
        block_errors[rk] = block_errors[rk][sort_idx]
        block_times[rk] = block_times[rk][sort_idx]
        block_error_bars[rk] = block_error_bars[rk][sort_idx]
        block_time_bars[rk] = block_time_bars[rk][sort_idx]
    end
    
    
    # Helper function to add error bands with CairoMakie
    function add_error_band!(ax, x_vals, y_vals, error_bars, valid_mask, color, alpha=0.3)
        if any(valid_mask)
            x_valid = x_vals[valid_mask]
            y_valid = y_vals[valid_mask]
            error_bars_valid = error_bars[valid_mask]
            
            # Extract lower and upper bounds
            lower_bounds = [eb[1] for eb in error_bars_valid]
            upper_bounds = [eb[2] for eb in error_bars_valid]
            
            # Add error band
            band!(ax, x_valid, lower_bounds, upper_bounds, 
                  color = (color, alpha))
        end
    end
    
    # Helper function to plot valid data points with CairoMakie
    function plot_valid_data!(ax, x_vals, y_vals, valid_mask, label, color, marker; linestyle=:solid)
        if any(valid_mask)
            x_valid = x_vals[valid_mask]
            y_valid = y_vals[valid_mask]
            lines!(ax, x_valid, y_valid, color = color, linewidth = 2, 
                   label = label, linestyle = linestyle)
            scatter!(ax, x_valid, y_valid, color = color, marker = marker, 
                    markersize = 8, strokewidth = 1, strokecolor = :white)
        end
    end
    
    # Accuracy comparison plot
    println("Creating accuracy comparison plot...")
    
    fig1 = Figure(size = (312, 254))
    ax1 = Axis(fig1[1, 1], 
              title = "Accuracy vs Target Rank",
              xlabel = "Target Rank r", 
              ylabel = "Relative Error",
              yscale = log10)
    
    # Filter out NaN values for plotting
    valid_det = .!isnan.(deterministic_errors)
    
    # Add error bands first (so they appear behind the lines)
    add_error_band!(ax1, target_ranks, deterministic_errors, deterministic_error_bars, valid_det, colors[:deterministic])
    for rk in all_block_rks
        valid = .!isnan.(block_errors[rk])
        add_error_band!(ax1, target_ranks, block_errors[rk], block_error_bars[rk], valid, colors[rk])
    end
    
    # Plot data with lines and markers
    plot_valid_data!(ax1, target_ranks, deterministic_errors, valid_det, 
                    "Deterministic (tt_rounding)", colors[:deterministic], markers[:deterministic])
    for rk in all_block_rks
        valid = .!isnan.(block_errors[rk])
        label = rk == 1 ? "R=1 (Khatri-Rao)" : "R=$rk"
        plot_valid_data!(ax1, target_ranks, block_errors[rk], valid, label, colors[rk], markers[rk])
    end
    
    # Add legend
    axislegend(ax1, position = :rt)
    
    accuracy_file = joinpath(save_dir, "hadamard_accuracy_comparison.pdf")
    save(accuracy_file, fig1)
    println("Accuracy plot saved: $accuracy_file")
    
    # Timing comparison plot
    println("Creating timing comparison plot...")
    
    fig2 = Figure(size = (312, 254))
    ax2 = Axis(fig2[1, 1], 
              title = "Timing vs Target Rank",
              xlabel = "Target Rank r", 
              ylabel = "Time (milliseconds)")
    
    # Filter out NaN values for timing plot
    valid_det_time = .!isnan.(deterministic_times)
    
    # Add timing error bands
    add_error_band!(ax2, target_ranks, deterministic_times, deterministic_time_bars, valid_det_time, colors[:deterministic])
    for rk in all_block_rks
        valid = .!isnan.(block_times[rk])
        add_error_band!(ax2, target_ranks, block_times[rk], block_time_bars[rk], valid, colors[rk])
    end
    
    # Plot timing data
    plot_valid_data!(ax2, target_ranks, deterministic_times, valid_det_time, 
                    "Deterministic (tt_rounding)", colors[:deterministic], markers[:deterministic])
    for rk in all_block_rks
        valid = .!isnan.(block_times[rk])
        label = rk == 1 ? "R=1 (Khatri-Rao)" : "R=$rk"
        plot_valid_data!(ax2, target_ranks, block_times[rk], valid, label, colors[rk], markers[rk])
    end
    
    # Add legend
    axislegend(ax2, position = :lt)
    
    timing_file = joinpath(save_dir, "hadamard_timing_comparison.pdf")
    save(timing_file, fig2)
    println("Timing plot saved: $timing_file")
    
    # Combined plot with subplots
    println("Creating combined comparison plot...")
    
    # Adjust figure size to accommodate horizontal legend below
    fig_combined = Figure(size = (624, 300))
    
    # Accuracy subplot
    ax_acc = Axis(fig_combined[1, 1], 
                 title = "Accuracy vs Target Rank",
                 xlabel = "Target Rank r", 
                 ylabel = "Relative Error",
                 yscale = log10)
    
    # Add error bands and data for accuracy
    add_error_band!(ax_acc, target_ranks, deterministic_errors, deterministic_error_bars, valid_det, colors[:deterministic])
    for rk in all_block_rks
        valid = .!isnan.(block_errors[rk])
        add_error_band!(ax_acc, target_ranks, block_errors[rk], block_error_bars[rk], valid, colors[rk])
    end
    
    plot_valid_data!(ax_acc, target_ranks, deterministic_errors, valid_det, 
                    "Deterministic", colors[:deterministic], markers[:deterministic])
    for rk in all_block_rks
        valid = .!isnan.(block_errors[rk])
        label = rk == 1 ? "R=1 (Khatri-Rao)" : "R=$rk"
        plot_valid_data!(ax_acc, target_ranks, block_errors[rk], valid, label, colors[rk], markers[rk])
    end
    
    # Timing subplot
    ax_time = Axis(fig_combined[1, 2], 
                  title = "Timing vs Target Rank",
                  xlabel = "Target Rank r", 
                  ylabel = "Time (milliseconds)")
    
    # Add error bands and data for timing
    add_error_band!(ax_time, target_ranks, deterministic_times, deterministic_time_bars, valid_det_time, colors[:deterministic])
    for rk in all_block_rks
        valid = .!isnan.(block_times[rk])
        add_error_band!(ax_time, target_ranks, block_times[rk], block_time_bars[rk], valid, colors[rk])
    end
    
    plot_valid_data!(ax_time, target_ranks, deterministic_times, valid_det_time, 
                    "Deterministic", colors[:deterministic], markers[:deterministic])
    for rk in all_block_rks
        valid = .!isnan.(block_times[rk])
        label = rk == 1 ? "R=1 (Khatri-Rao)" : "R=$rk"
        plot_valid_data!(ax_time, target_ranks, block_times[rk], valid, label, colors[rk], markers[rk])
    end
    
    # Add horizontal legend below both subplots
    Legend(fig_combined[2, 1:2], color_elements, color_labels,
            orientation = :horizontal, 
            tellheight = true, 
            framevisible = true,
            halign = :center,
            nbanks = 1,  # Single row
            padding = (4, 4, 2, 2),
            colgap = 8,
            labelsize = 9)
    
    combined_file = joinpath(save_dir, "hadamard_benchmark_comparison.pdf")
    save(combined_file, fig_combined)
    println("Combined plot saved: $combined_file")
    
    # Create timing breakdown plot
    println("Creating timing breakdown plot...")
    
    # Extract sketch and orthogonalization times with error bars
    sketch_times = Dict{Int, Vector{Float64}}()
    orthog_times = Dict{Int, Vector{Float64}}()
    sketch_error_bars = Dict{Int, Vector{Tuple{Float64, Float64}}}()
    orthog_error_bars = Dict{Int, Vector{Tuple{Float64, Float64}}}()

    for rk in all_block_rks
        sketch_times[rk] = Float64[]
        orthog_times[rk] = Float64[]
        sketch_error_bars[rk] = Tuple{Float64, Float64}[]
        orthog_error_bars[rk] = Tuple{Float64, Float64}[]
    end
    
    
    for (rank_key, rank_data) in results
        if startswith(rank_key, "target_rank_")
            for block_rks in all_block_rks
                block_key = "block_rks_$block_rks"
                if haskey(rank_data, block_key) && rank_data[block_key]["success_rate"] > 0
                    data = rank_data[block_key]
                    push!(sketch_times[block_rks], get(data, "median_sketch_time", 0.0) * 1000)  # Convert to ms
                    push!(orthog_times[block_rks], get(data, "median_orthog_time", 0.0) * 1000)  # Convert to ms
                    
                    # Calculate error bars for sketch and orthog times
                    if haskey(data, "sketch_times") && length(data["sketch_times"]) > 1
                        sketch_times_sorted = sort(data["sketch_times"]) * 1000  # Convert to ms
                        sketch_25 = quantile(sketch_times_sorted, 0.25)
                        sketch_75 = quantile(sketch_times_sorted, 0.75)
                        push!(sketch_error_bars[block_rks], (sketch_25, sketch_75))
                    else
                        median_sketch = get(data, "median_sketch_time", 0.0) * 1000
                        push!(sketch_error_bars[block_rks], (median_sketch, median_sketch))
                    end
                    
                    if haskey(data, "orthog_times") && length(data["orthog_times"]) > 1
                        orthog_times_sorted = sort(data["orthog_times"]) * 1000  # Convert to ms
                        orthog_25 = quantile(orthog_times_sorted, 0.25)
                        orthog_75 = quantile(orthog_times_sorted, 0.75)
                        push!(orthog_error_bars[block_rks], (orthog_25, orthog_75))
                    else
                        median_orthog = get(data, "median_orthog_time", 0.0) * 1000
                        push!(orthog_error_bars[block_rks], (median_orthog, median_orthog))
                    end
                else
                    push!(sketch_times[block_rks], NaN)
                    push!(orthog_times[block_rks], NaN)
                    push!(sketch_error_bars[block_rks], (NaN, NaN))
                    push!(orthog_error_bars[block_rks], (NaN, NaN))
                end
            end
        end
    end
    
    # Sort timing breakdown data
    for rk in all_block_rks
        sketch_times[rk] = sketch_times[rk][sort_idx]
        orthog_times[rk] = orthog_times[rk][sort_idx]
        sketch_error_bars[rk] = sketch_error_bars[rk][sort_idx]
        orthog_error_bars[rk] = orthog_error_bars[rk][sort_idx]
    end
    
    fig3 = Figure(size = (312, 254))
    ax3 = Axis(fig3[1, 1], 
              title = "Timing Breakdown",
              xlabel = "Target Rank r", 
              ylabel = "Time (milliseconds)")
    
    # Add error bands first (so they appear behind the lines)
    for rk in all_block_rks
        valid_sketch = .!isnan.(sketch_times[rk])
        valid_orthog = .!isnan.(orthog_times[rk])
        add_error_band!(ax3, target_ranks, sketch_times[rk], sketch_error_bars[rk], valid_sketch, colors[rk], 0.15)
        add_error_band!(ax3, target_ranks, orthog_times[rk], orthog_error_bars[rk], valid_orthog, colors[rk], 0.15)
    end
    
    # Plot sketch times (dashed lines)
    for rk in all_block_rks
        valid = .!isnan.(sketch_times[rk])
        label = rk == 1 ? "R=1 (sketch)" : "R=$rk (sketch)"
        plot_valid_data!(ax3, target_ranks, sketch_times[rk], valid, label, colors[rk], markers[rk]; linestyle=:dash)
    end
    
    # Plot orthogonalization times (solid lines)
    for rk in all_block_rks
        valid = .!isnan.(orthog_times[rk])
        label = rk == 1 ? "R=1 (QR)" : "R=$rk (QR)"
        plot_valid_data!(ax3, target_ranks, orthog_times[rk], valid, label, colors[rk], markers[rk])
    end
    
    # Add legend
    axislegend(ax3, position = :lt)
    
    breakdown_file = joinpath(save_dir, "hadamard_timing_breakdown.pdf")
    save(breakdown_file, fig3)
    println("Timing breakdown plot saved: $breakdown_file")

    # Memory usage comparison plot
    println("Creating memory usage comparison plot...")

    # Extract memory usage data
    deterministic_memory = Float64[]
    block_memory = Dict{Int, Vector{Float64}}()
    for rk in all_block_rks
        block_memory[rk] = Float64[]
    end

    # Error bar data for memory (25th and 75th percentiles)
    deterministic_memory_bars = Tuple{Float64, Float64}[]
    block_memory_bars = Dict{Int, Vector{Tuple{Float64, Float64}}}()
    for rk in all_block_rks
        block_memory_bars[rk] = Tuple{Float64, Float64}[]
    end

    for (rank_key, rank_data) in results
        if startswith(rank_key, "target_rank_")
            # Extract deterministic memory data
            if haskey(rank_data, "deterministic") && rank_data["deterministic"]["success_rate"] > 0
                det_data = rank_data["deterministic"]
                push!(deterministic_memory, det_data["median_memory"] / 1024^2)  # Convert to MB

                # Calculate error bars
                if haskey(det_data, "memory_usage") && length(det_data["memory_usage"]) > 1
                    memory_sorted = sort(det_data["memory_usage"]) / 1024^2  # Convert to MB
                    memory_25 = quantile(memory_sorted, 0.25)
                    memory_75 = quantile(memory_sorted, 0.75)
                    push!(deterministic_memory_bars, (memory_25, memory_75))
                else
                    median_mem = det_data["median_memory"] / 1024^2
                    push!(deterministic_memory_bars, (median_mem, median_mem))
                end
            else
                push!(deterministic_memory, NaN)
                push!(deterministic_memory_bars, (NaN, NaN))
            end

            # Extract memory data for each block configuration
            for block_rks in all_block_rks
                block_key = "block_rks_$block_rks"
                if haskey(rank_data, block_key) && rank_data[block_key]["success_rate"] > 0
                    block_data = rank_data[block_key]
                    push!(block_memory[block_rks], block_data["median_memory"] / 1024^2)  # Convert to MB

                    # Calculate error bars
                    if haskey(block_data, "memory_usage") && length(block_data["memory_usage"]) > 1
                        memory_sorted = sort(block_data["memory_usage"]) / 1024^2  # Convert to MB
                        memory_25 = quantile(memory_sorted, 0.25)
                        memory_75 = quantile(memory_sorted, 0.75)
                        push!(block_memory_bars[block_rks], (memory_25, memory_75))
                    else
                        median_mem = block_data["median_memory"] / 1024^2
                        push!(block_memory_bars[block_rks], (median_mem, median_mem))
                    end
                else
                    push!(block_memory[block_rks], NaN)
                    push!(block_memory_bars[block_rks], (NaN, NaN))
                end
            end
        end
    end

    # Sort memory data
    deterministic_memory = deterministic_memory[sort_idx]
    deterministic_memory_bars = deterministic_memory_bars[sort_idx]

    for rk in all_block_rks
        block_memory[rk] = block_memory[rk][sort_idx]
        block_memory_bars[rk] = block_memory_bars[rk][sort_idx]
    end

    fig4 = Figure(size = (312, 254))
    ax4 = Axis(fig4[1, 1], 
              title = "Memory Usage vs Target Rank",
              xlabel = "Target Rank r", 
              ylabel = "Memory Usage (MB)")

    # Filter out NaN values for memory plot
    valid_det_memory = .!isnan.(deterministic_memory)

    # Add memory error bands first
    add_error_band!(ax4, target_ranks, deterministic_memory, deterministic_memory_bars, valid_det_memory, colors[:deterministic])
    for rk in all_block_rks
        valid = .!isnan.(block_memory[rk])
        add_error_band!(ax4, target_ranks, block_memory[rk], block_memory_bars[rk], valid, colors[rk])
    end

    # Plot memory data
    plot_valid_data!(ax4, target_ranks, deterministic_memory, valid_det_memory, 
                    "Deterministic (tt_rounding)", colors[:deterministic], markers[:deterministic])
    for rk in all_block_rks
        valid = .!isnan.(block_memory[rk])
        label = rk == 1 ? "R=1 (Khatri-Rao)" : "R=$rk"
        plot_valid_data!(ax4, target_ranks, block_memory[rk], valid, label, colors[rk], markers[rk])
    end

    # Add legend
    axislegend(ax4, position = :lt)

    memory_file = joinpath(save_dir, "hadamard_memory_comparison.pdf")
    save(memory_file, fig4)
    println("Memory comparison plot saved: $memory_file")

    return accuracy_file, timing_file, combined_file, breakdown_file, memory_file
end

"""
Load benchmark results from JSON file if it exists
"""
function load_benchmark_results(dir="out/hadamard_benchmarks")
    results_file = joinpath(dir, "hadamard_benchmark_results.json")
    
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
                               dir="out/hadamard_benchmarks")
    mkpath(dir)
    
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
    
    results_file = joinpath(dir, "hadamard_benchmark_results.json")
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
    ###final_results = analyze_sketching_results(results)
    final_results = nothing
    # Step 7: Create plots
    println("\n" * "="^60)
    println("CREATING VISUALIZATION PLOTS")
    println("="^60)

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
function create_plots_from_existing_results(dir="out/hadamard_benchmarks")
    println("Loading existing results and creating plots...")
    
    # Load existing results
    results, metadata = load_benchmark_results(dir)
    
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
    plot_files = create_hadamard_benchmark_plots(results, dir)
    
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
        target_ranks=[16, 32, 48, 64, 80, 96, 112, 128, 144, 160],
        block_rks_options=[1, 4, 8, 16], 
        n_trials=100
    )
end

global results, ttvectors, exact_solution = run_quantics_hadamard_benchmark(
        R=20,  # Smaller scale for testing
        target_ranks=[16, 32, 48, 64, 80, 96, 112, 128, 144, 160],
        block_rks_options=[1, 4, 8, 16], 
        n_trials=100
    )
