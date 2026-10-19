using AdaptiveCrossApproximation
using LinearAlgebra
using Random
using Test

@testset "combinedconvcrit.jl" begin
    @testset "CombinedConvCritFunctor" begin
        K = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 10.0]
        rowidcs = [1, 2, 3]
        colidcs = [1, 2, 3]

        Random.seed!(7)
        combined = AdaptiveCrossApproximation.CombinedConvCrit([
            AdaptiveCrossApproximation.FNormEstimator(0.3),
            AdaptiveCrossApproximation.RandomSampling(; nsamples=3, tol=0.3),
        ],)

        functor = combined(K, rowidcs, colidcs)
        @test length(functor.crits) == 2

        rowbuffer = zeros(1, 3)
        colbuffer = zeros(3, 1)
        rowbuffer[1, :] .= [1.0, 0.0, 0.0]
        colbuffer[:, 1] .= [1.0, 1.0, 0.0]

        npivot, conv = functor(rowbuffer, colbuffer, 1, 3, 3)
        @test npivot == 1
        @test conv

        reset!(functor, [1, 2], [1, 2, 3])
        @test iszero(functor.crits[1].normUV)
        @test iszero(functor.crits[2].normUV)
        @test functor.crits[2].nactive == 3
        @test length(functor.crits[2].indices) >= 3
        @test length(functor.crits[2].rest) >= 3
    end
end
