using Revise
using TensorTrains
using LinearAlgebra
using Random
using Printf
using QuanticsTCI
using Plots
using JSON3
using Statistics
using TimerOutputs

"""
QuanticsTCI Rounding Algorithm Benchmark

Compare deterministic (tt_rounding) vs randomized (ttrand_rounding) algorithms
using QuanticsTCI-generated tensor trains as realistic test cases.
"""

include("quantics_example.jl")

"""
Simple benchmark comparing tt_rounding vs ttrand_rounding
"""
function benchmark_rounding_algorithms(ttvs::NTuple{3,TTvector{T,N}}; 
    max_ranks = [8, 16, 32, 64, 128],
    n_trials = 3,
    seed = 1234
) where {T,N}
    
    println("=== Rounding Algorithm Benchmark ===")
    println("TTvectors: $(N) modes, original max rank: $(maximum(maximum(ttv.ttv_rks) for ttv in ttvs))")
    println("Max ranks to test: $(max_ranks)")
    println()
    
    # Compute original norm for error calculation
    ttv = ttvs[3]*ttvs[2]*ttvs[1]
    original_norm = norm(ttv)
    ttv_exact = tt_rounding(ttv, tol=1e-12)

    println("Original norm: $(@sprintf "%.6e" original_norm)")
    println()
    
    results = Dict{String, Any}()
    
    # Test different max rank constraints
    for rmax in max_ranks
        
        println("=== Testing rmax = $rmax ===")
        rmax_results = Dict{String, Any}("rmax" => rmax)
        
        # Test tt_rounding with max rank
        print("  tt_rounding: ")
        
        det_times = Float64[]
        det_errors = Float64[]
        det_ranks = Vector{Int}[]
        

        # Warm start
        tt_rounding(ttv; rmax=rmax)

        for trial in 1:n_trials
            
            try
                start_time = time()
                ttv_compressed = tt_rounding(ttv; rmax=rmax)
                comp_time = time() - start_time
                
                # Compute error
                error_norm = norm(ttv_exact - ttv_compressed)
                
                rel_error = error_norm / original_norm
                
                push!(det_times, comp_time)
                push!(det_errors, rel_error)
                push!(det_ranks, ttv_compressed.ttv_rks)
                
                print("$(@sprintf "%.3f" comp_time)s ")
                
            catch e
                print("FAILED ")
            end
        end
        
        println()
        
        rmax_results["tt_rounding"] = Dict(
            "times" => det_times,
            "errors" => det_errors,
            "ranks" => det_ranks,
            "mean_time" => isempty(det_times) ? NaN : mean(det_times),
            "mean_error" => isempty(det_errors) ? NaN : mean(det_errors),
            "mean_max_rank" => isempty(det_ranks) ? NaN : mean([maximum(rks) for rks in det_ranks])
        )
        
        # Test ttrand_rounding with target ranks
        print("  ttrand_rounding: ")
        
        rand_times = Float64[]
        rand_errors = Float64[]
        rand_ranks = Vector{Int}[]
        
        # Create target rank structure
        target_rks = min.(rmax, ttv.ttv_rks)
        
        # Warm start
        ttrand_rounding(ttvs, target_rks, seed=1234, block_rks=1)


                to = TimerOutput()
        for trial in 1:n_trials            
            try
                start_time = time()
                ttv_rand = ttrand_rounding(ttvs, target_rks, block_rks=1, seed=seed + trial, timer=to)
                rand_time = time() - start_time
                
                # Compute error
                error_norm = norm(ttv_exact - ttv_rand)
                
                rel_error = error_norm / original_norm
                
                push!(rand_times, rand_time)
                push!(rand_errors, rel_error)
                push!(rand_ranks, ttv_rand.ttv_rks)
                
                print("$(@sprintf "%.3f" rand_time)s ")
                
            catch e
                print("FAILED ")
            end
        end
                display(to)
        
        println()
        
        rmax_results["ttrand_rounding"] = Dict(
            "target_rks" => target_rks,
            "times" => rand_times,
            "errors" => rand_errors,
            "ranks" => rand_ranks,
            "mean_time" => isempty(rand_times) ? NaN : mean(rand_times),
            "mean_error" => isempty(rand_errors) ? NaN : mean(rand_errors),
            "mean_max_rank" => isempty(rand_ranks) ? NaN : mean([maximum(rks) for rks in rand_ranks])
        )
        
        # Print comparison for this rmax
        println("  Results summary:")
        det_data = rmax_results["tt_rounding"]
        rand_data = rmax_results["ttrand_rounding"]
        
        if !isnan(det_data["mean_error"]) && !isnan(det_data["mean_time"])
            println("    tt_rounding: error=$(@sprintf "%.2e" det_data["mean_error"]), time=$(@sprintf "%.3f" det_data["mean_time"])s")
        end
        
        if !isnan(rand_data["mean_error"]) && !isnan(rand_data["mean_time"])
            println("    ttrand_rounding: error=$(@sprintf "%.2e" rand_data["mean_error"]), time=$(@sprintf "%.3f" rand_data["mean_time"])s")
        end
        
        println()
        
        results["rmax_$(rmax)"] = rmax_results
    end
    
    return results
end

"""
Run benchmarks on different QuanticsTCI test cases
"""
function run_quantics_benchmarks()
    println("=== QuanticsTCI Rounding Algorithm Benchmarks ===")
    println()
    
    all_results = Dict{String, Any}()
    
    # Test Case 1: Medium scale
    println("=== Benchmark 1: Medium Scale (R=10) ===")
    qtt_medium, ttv_medium = run_quantics_example(R=10, tolerance=1e-9, test_points=10)
    
    results_medium = benchmark_rounding_algorithms(
        ttv_medium; 
        max_ranks=[8, 16, 32, 64, 128],
        n_trials=10
    )
    all_results["medium_R10"] = results_medium
    
    println("\\n" * "="^80 * "\\n")
    
    # Test Case 2: Large scale
    println("=== Benchmark 2: Large Scale (R=15) ===")
    qtt_large, ttv_large = run_quantics_example(R=15, tolerance=1e-9, test_points=10)
    
    results_large = benchmark_rounding_algorithms(
        ttv_large; 
        max_ranks=[8, 16, 32, 64, 128],
        n_trials=10
    )
    all_results["large_R20"] = results_large
    
    # Save results
    mkpath("out/quantics_benchmarks")
    results_file = "out/quantics_benchmarks/rounding_benchmarks.json"
    open(io -> JSON3.write(io, all_results, allow_inf=true), results_file, "w")
    println("Results saved to: $results_file")
    
    # Create summary plots
    create_benchmark_plots(all_results)
    
    return all_results
end

"""
Create plots comparing the algorithms
"""
function create_benchmark_plots(all_results)
    println("Creating benchmark plots...")
    
    mkpath("out/quantics_benchmarks/plots")
    
    for (test_case, case_results) in all_results
        try
            # Extract data for plotting
            rmax_values = Int[]
            ttrand_errors = Float64[]
            ttrand_times = Float64[]
            tt_errors = Float64[]  # Best tt_rounding result
            tt_times = Float64[]
            
            for (rmax_key, rmax_data) in case_results
                if startswith(rmax_key, "rmax_")
                    rmax = parse(Int, replace(rmax_key, "rmax_" => ""))
                    push!(rmax_values, rmax)
                    
                    # ttrand_rounding data
                    rand_data = rmax_data["ttrand_rounding"]
                    push!(ttrand_errors, rand_data["mean_error"])
                    push!(ttrand_times, rand_data["mean_time"])
                    
                    # tt_rounding data
                    det_data = rmax_data["tt_rounding"]
                    push!(tt_errors, det_data["mean_error"])
                    push!(tt_times, det_data["mean_time"])
                end
            end
            
            # Sort by rmax
            sort_idx = sortperm(rmax_values)
            rmax_values = rmax_values[sort_idx]
            ttrand_errors = ttrand_errors[sort_idx]
            ttrand_times = ttrand_times[sort_idx]
            tt_errors = tt_errors[sort_idx]
            tt_times = tt_times[sort_idx]
            
            # Error comparison plot
            if !all(isnan.(ttrand_errors)) && !all(isinf.(tt_errors))
                p1 = plot(title="Compression Error vs Max Rank ($test_case)", 
                         xlabel="Max Rank", ylabel="Relative Error", 
                         yscale=:log10, legend=:topright)
                
                plot!(p1, rmax_values, ttrand_errors, label="ttrand_rounding", 
                     marker=:circle, linewidth=2, color=:blue)
                plot!(p1, rmax_values, tt_errors, label="tt_rounding", 
                     marker=:square, linewidth=2, color=:red)
                
                error_file = "out/quantics_benchmarks/plots/error_comparison_$(test_case).png"
                savefig(p1, error_file)
                println("Error plot saved: $error_file")
            end
            
            # Time comparison plot
            if !all(isnan.(ttrand_times)) && !all(isnan.(tt_times))
                p2 = plot(title="Computation Time vs Max Rank ($test_case)", 
                         xlabel="Max Rank", ylabel="Time (seconds)", 
                         legend=:topright)
                
                plot!(p2, rmax_values, ttrand_times, label="ttrand_rounding", 
                     marker=:circle, linewidth=2, color=:blue)
                plot!(p2, rmax_values, tt_times, label="tt_rounding", 
                     marker=:square, linewidth=2, color=:red)
                
                time_file = "out/quantics_benchmarks/plots/time_comparison_$(test_case).png"
                savefig(p2, time_file)
                println("Time plot saved: $time_file")
            end
            
        catch e
            println("Warning: Could not create plots for $test_case: $e")
        end
    end
end

"""
Print summary of benchmark results
"""
function print_benchmark_summary(all_results)
    println("\\n=== Benchmark Summary ===")
    
    for (test_case, case_results) in all_results
        println("\\n--- $test_case ---")
        
        total_ttrand_speedup = Float64[]
        total_accuracy_ratio = Float64[]
        
        for (rmax_key, rmax_data) in case_results
            if startswith(rmax_key, "rmax_")
                rmax = parse(Int, replace(rmax_key, "rmax_" => ""))
                
                det_data = rmax_data["tt_rounding"]
                rand_data = rmax_data["ttrand_rounding"]
                
                det_time = det_data["mean_time"]
                det_error = det_data["mean_error"]
                
                if !isnan(rand_data["mean_time"]) && !isnan(det_time)
                    speedup = det_time / rand_data["mean_time"]
                    push!(total_ttrand_speedup, speedup)
                end
                
                if !isnan(rand_data["mean_error"]) && !isnan(det_error)
                    accuracy_ratio = rand_data["mean_error"] / det_error
                    push!(total_accuracy_ratio, accuracy_ratio)
                end
                
                println("  rmax=$rmax:")
                println("    tt_rounding: error=$(@sprintf "%.2e" det_error), time=$(@sprintf "%.3f" det_time)s")
                println("    ttrand_rounding: error=$(@sprintf "%.2e" rand_data["mean_error"]), time=$(@sprintf "%.3f" rand_data["mean_time"])s")
            end
        end
        
        if !isempty(total_ttrand_speedup)
            println("  Average speedup (tt_time/ttrand_time): $(@sprintf "%.2f" mean(total_ttrand_speedup))")
        end
        if !isempty(total_accuracy_ratio)
            println("  Average accuracy ratio (ttrand_error/tt_error): $(@sprintf "%.2f" mean(total_accuracy_ratio))")
        end
    end
end

# Run benchmarks if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    Random.seed!(1234)
    results = run_quantics_benchmarks()
    print_benchmark_summary(results)
end