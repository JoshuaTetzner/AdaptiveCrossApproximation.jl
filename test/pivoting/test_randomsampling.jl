using AdaptiveCrossApproximation
using LinearAlgebra
using Random
using StaticArrays
using Test

@testset "randomsampling.jl" begin
    @testset "RandomSamplingPivoting" begin
        K = reshape(collect(1.0:12.0), 3, 4)
        rowidcs = [1, 2, 3]
        colidcs = [1, 2, 3, 4]

        Random.seed!(11)
        cc = AdaptiveCrossApproximation.RandomSampling(; nsamples=5, tol=0.2)
        convcrit = cc(K, rowidcs, colidcs)

        rowpiv = AdaptiveCrossApproximation.RandomSamplingPivoting(1)(convcrit)
        colpiv = AdaptiveCrossApproximation.RandomSamplingPivoting(2)(convcrit)
        @test rowpiv.rc == 1
        @test colpiv.rc == 2
        @test resize!(rowpiv) === nothing
        @test reset!(rowpiv) === nothing

        rowbuffer = zeros(2, 4)
        colbuffer = zeros(3, 2)
        rowbuffer[1, :] .= [1.0, 2.0, 0.0, 0.0]
        colbuffer[:, 1] .= [1.0, 0.0, 1.0]
        convcrit(rowbuffer, colbuffer, 1, 3, 4)

        expected = convcrit.indices[argmax(abs.(view(convcrit.rest, 1:(convcrit.nactive))))]
        @test rowpiv(Float64[]) == expected[1]
        @test colpiv(Float64[]) == expected[2]

        combined = AdaptiveCrossApproximation.CombinedConvCritFunctor(
            AdaptiveCrossApproximation.ConvCritFunctor[convcrit], [true]
        )
        piv2 = AdaptiveCrossApproximation.RandomSamplingPivoting(1)(combined)
        @test piv2(Float64[]) == expected[1]

        badcombined = AdaptiveCrossApproximation.CombinedConvCritFunctor(
            AdaptiveCrossApproximation.ConvCritFunctor[AdaptiveCrossApproximation.FNormEstimator(
                1e-4
            )()],
            [true],
        )
        @test_throws ArgumentError AdaptiveCrossApproximation.RandomSamplingPivoting(1)(
            badcombined
        )
    end
end
