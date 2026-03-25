using AdaptiveCrossApproximation
using LinearAlgebra
using Random
using StaticArrays
using Test

@testset "ACA" begin
    Random.seed!(1234)

    tpos = [@SVector rand(3) for _ in 1:48]
    spos = [@SVector rand(3) for _ in 1:52] .+ Scalar(SVector(3.5, 0.0, 0.0))
    Kc = [inv(norm(tp - sp)) for tp in tpos, sp in spos]

    rowpivotings = [
        AdaptiveCrossApproximation.MaximumValue(), AdaptiveCrossApproximation.Leja2(tpos)
    ]
    colpivotings = [
        AdaptiveCrossApproximation.MaximumValue(),
        AdaptiveCrossApproximation.Leja2(spos),
        AdaptiveCrossApproximation.FillDistance(spos),
    ]
    mk_convergence = [
        AdaptiveCrossApproximation.FNormEstimator(1e-4),
        AdaptiveCrossApproximation.FNormExtrapolator(1e-4),
        AdaptiveCrossApproximation.RandomSampling(; tol=1e-4, nsamples=120),
        AdaptiveCrossApproximation.CombinedConvCrit([
            AdaptiveCrossApproximation.FNormEstimator(1e-4),
            AdaptiveCrossApproximation.RandomSampling(; tol=1e-4, nsamples=120),
        ]),
    ]

    for rowpivoting in rowpivotings
        for colpivoting in colpivotings
            for convergence in mk_convergence
                U, V = AdaptiveCrossApproximation.aca(
                    Kc;
                    rowpivoting=rowpivoting,
                    columnpivoting=colpivoting,
                    convergence=convergence,
                    maxrank=30,
                )

                @test size(U, 1) == size(Kc, 1)
                @test size(V, 2) == size(Kc, 2)
                @test size(U, 2) == size(V, 1)
                @test norm(U * V - Kc) / norm(Kc) < 2e-4
            end
        end
    end
end

@testset "ACA Special Cases" begin
    Random.seed!(1234)
    K = zeros(10, 10)
    U, V = AdaptiveCrossApproximation.aca(K; tol=10^-4, maxrank=5)
    @test length(U) == 0
    @test length(V) == 0

    K[4, :] = rand(10)
    U, V = AdaptiveCrossApproximation.aca(K; tol=10^-4, maxrank=5)
    @test size(U, 2) == 1
    @test size(V, 1) == 1

    K[1:2, :] = rand(2, 10)
    U, V = AdaptiveCrossApproximation.aca(K; tol=10^-4, maxrank=5)
    @test size(U, 2) == 3
    @test size(V, 1) == 3
end
