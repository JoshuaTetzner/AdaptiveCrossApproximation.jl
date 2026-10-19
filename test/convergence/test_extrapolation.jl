using AdaptiveCrossApproximation
using LinearAlgebra
using Random
using Test

@testset "extrapolation.jl" begin
    @testset "FNormExtrapolatorFunctor" begin
        cc = AdaptiveCrossApproximation.FNormExtrapolator(0.2)
        functor = cc(6)

        rowbuffer = [1.0 0.0; 0.0 0.0]
        colbuffer = [1.0 0.0; 0.0 0.0]

        npivot1, conv1 = functor(rowbuffer, colbuffer, 1, 2, 2)
        @test npivot1 == 1
        @test conv1
        @test isapprox(functor.lastnorms[1], 1.0)

        npivot2, conv2 = functor(rowbuffer, colbuffer, 2, 2, 2)
        @test npivot2 == 1
        @test !conv2

        reset!(functor)
        @test all(iszero, functor.lastnorms)
        @test iszero(functor.estimator.normUV)

        cc_iaca = AdaptiveCrossApproximation.FNormExtrapolator(
            AdaptiveCrossApproximation.FNormEstimator(0.5)
        )
        functor_iaca = cc_iaca(5)
        buf = [3.0, 4.0]
        AdaptiveCrossApproximation.normF!(functor_iaca.estimator, buf, 1)
        npivotv, convv = functor_iaca(buf, 1)
        @test npivotv == 1
        @test convv
        @test isapprox(functor_iaca.lastnorms[1], 5.0)
    end
end
