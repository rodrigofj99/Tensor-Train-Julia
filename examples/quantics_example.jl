using Revise
using TensorTrains
using LinearAlgebra
using Random
using Printf
using QuanticsTCI

"""
Complete QuanticsTCI to TensorTrains Integration Example

This script demonstrates:
1. Defining a test function f(x,y,z) with interesting multi-scale structure
2. Using QuanticsTCI to interpolate the function on a quantics grid (2^R points per dimension)
3. Converting the resulting QTT object to TensorTrains TTvector format
4. Testing the conversion and demonstrating the workflow

The quantics representation uses binary encoding where each coordinate is represented
as a tensor train over {0,1} binary indices, allowing exponentially large grids
with polynomial-rank tensor representations.
"""

# Test function with interesting multi-scale structure: f = term1*term2*term3
function term1(x, y, z, R)
    t = x+y-2z
    return .1*exp(-2*(x^2 + y^2 + z^2)) * exp(-t^2) + exp(x)
end
function term2(x, y, z, R)
    B = 2.0^(4-R)
    t = x+y-2z
    return .1*cos(t / B) + exp(-x)
end
function term3(x, y, z, R)
    B = 2.0^(4-R)
    t = x+y-2z
    return .1*cos(t / (4 * sqrt(5) * B)) + 1
end

function test_function_3d(x, y, z, R)
    return term1(x,y,z,R)*term2(x,y,z,R)*term3(x,y,z,R)
end

"""
Convert a QuanticsTCI QuanticsTensorCI2 object to TensorTrains TTvector format.

This extracts the underlying TCI tensor cores and converts them to TTvector representation.
QuanticsTCI uses TensorCrossInterpolation internally which stores tensor train cores.
"""
function qtt_to_ttvector(qtt_obj::QuanticsTCI.QuanticsTensorCI2{T}) where {T}
    println("Converting QuanticsTCI object to TTvector format...")
    
    cores = qtt_obj.tci.sitetensors
    N = length(cores)

    ttv_cores = Vector{Array{T,3}}(undef, N)
    for i=1:N
        ttv_cores[i] = permutedims(cores[i], (2,1,3))
    end

    # Determine dimensions - for quantics, should be all binary
    dims = Tuple(size(cores[i], 2) for i in 1:N)
    println("Inferred dimensions: $dims")
    
    # Extract ranks from core sizes
    ranks = Vector{Int}(undef, N+1)
    for i in 1:N
        ranks[i] = size(cores[i], 1)
    end
    ranks[N+1] = size(cores[N],3)
    println("Extracted ranks: $ranks")
    
    # Create TTvector using the proper constructor
    ttvector = TTvector{T,N}(N, ttv_cores, dims, ranks, zeros(Int64, N))
    println("Successfully created TTvector!")
    
    return ttvector
end

"""
Run QuanticsTCI interpolation and convert to TensorTrains format
"""
function run_quantics_example(;
    R = 10,  # 2^R = 1024 grid points per dimension  
    tolerance = 1e-12,  # TCI tolerance
    test_points = 100,  # Number of random test points for accuracy check
    seed = 1234
)
    Random.seed!(seed)
    
    println("=== QuanticsTCI 3D Function Interpolation Example ===")
    println("Grid size: $(2^R)³ = $(2^(3*R)) total points")
    println("TCI tolerance: $tolerance")
    println()
    
    # Create coordinate arrays for each dimension
    xvals = range(0.0, 1.0; length=2^R)
    yvals = range(0.0, 1.0; length=2^R)
    zvals = range(0.0, 1.0; length=2^R)
    
    # Create 3-argument function wrapper
    f1(x, y, z) = term1(x, y, z, R)
    f2(x, y, z) = term2(x, y, z, R)
    f3(x, y, z) = term3(x, y, z, R)
    f = [f1,f2,f3]
    
    println("Starting QuanticsTCI interpolation...")
    
    # Use QuanticsTCI API
    interpolants = ntuple(j->quanticscrossinterpolate(
        Float64, 
        f[j], 
        [xvals, yvals, zvals]; 
        tolerance=tolerance,
        unfoldingscheme=:interleaved ), 3)
    qtt_interpolants = getindex.(interpolants,1)
    ranks = getindex.(interpolants,2)
    errors = getindex.(interpolants,3)

    qtt_interpolant(ix, iy, iz) = 
        qtt_interpolants[1](ix,iy,iz)*qtt_interpolants[2](ix,iy,iz)*qtt_interpolants[3](ix,iy,iz)
    
    println("QuanticsTCI interpolation completed!")
    
    # Test accuracy on random points
    println("\n=== Accuracy Test ===")
    
    test_errors = Float64[]
    
    for i in 1:test_points
        # Random test point in [0,1]³
        x_test = rand()
        y_test = rand() 
        z_test = rand()
        
        # Exact function value
        exact_value = test_function_3d(x_test, y_test, z_test, R)
        
        # Convert continuous coordinates to grid indices
        ix = clamp(round(Int, x_test * (2^R - 1)) + 1, 1, 2^R)
        iy = clamp(round(Int, y_test * (2^R - 1)) + 1, 1, 2^R)
        iz = clamp(round(Int, z_test * (2^R - 1)) + 1, 1, 2^R)
        
        # Interpolated value using discrete indices
        interpolated_value = qtt_interpolant(ix, iy, iz)
        
        # Get the actual grid point coordinates for fair comparison
        x_grid = xvals[ix]
        y_grid = yvals[iy]
        z_grid = zvals[iz]
        exact_at_grid = test_function_3d(x_grid, y_grid, z_grid, R)
        
        error = abs(exact_at_grid - interpolated_value)
        push!(test_errors, error)
    end
    
    if !isempty(test_errors)
        mean_error = sum(test_errors) / length(test_errors)
        max_error = maximum(test_errors)
        
        println("Mean absolute error: $(@sprintf "%.2e" mean_error)")
        println("Maximum absolute error: $(@sprintf "%.2e" max_error)")
        println("Relative accuracy: $(@sprintf "%.2e" max_error / maximum(abs(test_function_3d(rand(), rand(), rand(), R)) for _ in 1:100))")
    end
    
    println("\n=== Conversion to TensorTrains Format ===")
    
    # Convert QTT to TensorTrains TTvector format
    ttvectors = qtt_to_ttvector.(qtt_interpolants)
    
    println("Successfully converted to TTvector!")
    println("TT dimensions: $(ttvectors[1].ttv_dims)")
    println("TT ranks: $(broadcast(*, getproperty.(ttvectors, :ttv_rks)...))")
    println("Total parameters: $(sum(sum(prod(size(core)) for core in ttvector.ttv_vec) for ttvector in ttvectors))")
    
    # Test TensorTrains operations - skip problematic operations for high-rank tensors
    println("\n=== TensorTrains Operations ===")
    println("✓ TTvectors successfully created with $(length(ttvectors[1].ttv_dims)) binary modes")
    println("✓ Maximum rank: $(maximum(maximum(ttvector.ttv_rks) for ttvector in ttvectors))")
    
    println("✓ Testing compression...")
    ttv_compressed = tt_rounding(ttvectors[1]*ttvectors[2]*ttvectors[3], tol=1e-6)
    println("✓ Compression: $(maximum(broadcast(*, getproperty.(ttvectors, :ttv_rks)...))) → $(maximum(ttv_compressed.ttv_rks))")
    
    return qtt_interpolants, ttvectors
end

# Examples with different scales
function run_examples()
    println("=== QuanticsTCI to TensorTrains Integration Examples ===\n")
    
    # Example 1: Small scale for testing
    println("=== Example 1: Small Scale (R=4, 16³ grid) ===")
    qtt_small, ttv_small = run_quantics_example(R=4, tolerance=1e-6, test_points=20)
    
    println("\n" * "="^60 * "\n")
    
    # Example 2: Medium scale  
    println("=== Example 2: Medium Scale (R=6, 64³ grid) ===")
    qtt_medium, ttv_medium = run_quantics_example(R=6, tolerance=1e-8, test_points=50)
    
    println("\n" * "="^60 * "\n")
    
    # Example 3: Large scale
    println("=== Example 3: Large Scale (R=10, 1024³ grid) ===")
    qtt_large, ttv_large = run_quantics_example(R=10, tolerance=1e-12, test_points=100)
    
    println("\n" * "="^60 * "\n")
    println("All examples completed successfully!")
    println("This demonstrates the complete workflow for using QuanticsTCI.jl with TensorTrains.jl")
    
    return qtt_small, ttv_small, qtt_medium, ttv_medium, qtt_large, ttv_large
end

# Run examples if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_examples()
end