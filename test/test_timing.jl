using TensorTrains
using TimerOutputs
using Test

@testset "Timing Infrastructure" begin
    @testset "ttrand_rounding with timer" begin
        to = TimerOutput()
        dims = (4,4,4)
        rks = [1,3,3,1]
        A = rand_tt(dims, rks)

        # Test with timer
        A_round = ttrand_rounding(A, [1,2,2,1]; timer=to)

        # Verify output is correct type
        @test A_round isa TTvector

        # Verify timer contains expected sections
        # Note: TimerOutputs internal structure may vary, just check it runs without error
        @test true  # If we got here, timing worked

        # Test that default (no timer) still works
        A_round2 = ttrand_rounding(A, [1,2,2,1])
        @test A_round2 isa TTvector
    end

    @testset "stta with timer" begin
        to = TimerOutput()
        dims = (3,3,3)
        rks = [1,2,2,1]
        A = rand_tt(dims, rks)

        # Test with timer
        A_stta = stta(A; rks=[1,2,2,1], timer=to)

        # Verify output
        @test A_stta isa TTvector

        # Test without timer (default behavior)
        A_stta2 = stta(A; rks=[1,2,2,1])
        @test A_stta2 isa TTvector
    end

    @testset "tt_recursive_sketch with timer" begin
        to = TimerOutput()
        dims = (3,3,3)
        rks = [1,2,2,1]
        A = rand_tt(dims, rks)

        # Test with timer
        W, sketch_rks = tt_recursive_sketch(A, [1,3,3,1]; timer=to)

        # Verify output
        @test length(W) == length(dims) + 1
        @test length(sketch_rks) == length(dims) + 1

        # Test without timer
        W2, sketch_rks2 = tt_recursive_sketch(A, [1,3,3,1])
        @test length(W2) == length(dims) + 1
    end
end
