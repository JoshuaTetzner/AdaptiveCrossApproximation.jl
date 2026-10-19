using AdaptiveCrossApproximation
using LinearAlgebra
using Random
using Test

@testset "estimation.jl" begin
    @testset "FNormEstimatorFunctor" begin
        cc = AdaptiveCrossApproximation.FNormEstimator(0.5)
        functor = cc()

        rowbuffer = [1.0 0.0 0.0; 0.0 2.0 0.0]
        colbuffer = [3.0 0.0; 0.0 4.0]

        npivot1, conv1 = functor(rowbuffer, colbuffer, 1, 2, 3)
        @test npivot1 == 1
        @test conv1
        @test isapprox(functor.normUV, 3.0)
        @test AdaptiveCrossApproximation.tolerance(cc) == 0.5

        npivot2, conv2 = functor(rowbuffer, colbuffer, 2, 2, 3)
        @test npivot2 == 2
        @test conv2
        @test isapprox(functor.normUV, sqrt(73))

        rowzero = zeros(1, 2)
        colnonzero = ones(1, 1)
        np0, conv0 = functor(rowzero, colnonzero, 1, 1, 2)
        @test np0 == 0
        @test conv0

        bothzero = zeros(1, 1)
        npb, convb = functor(bothzero, bothzero, 1, 1, 1)
        @test npb == 0
        @test !convb

        reset!(functor)
        @test iszero(functor.normUV)
    end

    @testset "FNormEstimatorFunctor (iACA)" begin
        cc = AdaptiveCrossApproximation.FNormEstimator(0.8)
        functor = cc()

        buf1 = [3.0, 4.0]
        AdaptiveCrossApproximation.normF!(functor, buf1, 1)
        @test isapprox(functor.normUV, 5.0)
        @test AdaptiveCrossApproximation.tolerance(cc) == 0.8

        npivot1, conv1 = functor(buf1, 1)
        @test npivot1 == 1
        @test conv1

        buf2 = [0.0, 4.0]
        AdaptiveCrossApproximation.normF!(functor, buf2, 2)
        @test isapprox(functor.normUV, 4.5)

        npivot2, conv2 = functor(zeros(2), 2)
        @test npivot2 == 1
        @test !conv2

        reset!(functor)
        @test iszero(functor.normUV)
    end
end
