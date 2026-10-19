using AdaptiveCrossApproximation
using LinearAlgebra
using Random
using StaticArrays
using Test

@testset "combinedpivstrat.jl" begin
    @testset "CombinedPivStrat" begin
        pts = [SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0), SVector(2.0, 0.0, 0.0)]
        strats = [
            AdaptiveCrossApproximation.MaximumValue(), AdaptiveCrossApproximation.Leja2(pts)
        ]
        comb = AdaptiveCrossApproximation.CombinedPivStrat(strats)

        conv = AdaptiveCrossApproximation.CombinedConvCritFunctor(
            [
                AdaptiveCrossApproximation.FNormEstimator(1e-4)(),
                AdaptiveCrossApproximation.FNormEstimator(1e-4)(),
            ],
            [true, false],
        )

        functor = comb(conv, [1, 2, 3])

        idx1 = functor([0.1, 2.0, 0.2])
        @test idx1 == 2

        conv.isconverged .= [false, true]
        idx2 = functor([0.1, 2.0, 0.2])
        @test idx2 == 1

        reset!(functor, [3, 1, 2, 3, 1])
        @test all(.!view(functor.strats[1].usedidcs, 1:5))
        @test collect(view(functor.strats[2].idcs, 1:5)) == [3, 1, 2, 3, 1]
        @test all(iszero, view(functor.strats[2].h, 1:5))

        pts_fd = [@SVector [Float64(i), 0.0, 0.0] for i in 1:6]
        strats_fd = [
            AdaptiveCrossApproximation.MaximumValue(),
            AdaptiveCrossApproximation.FillDistance(pts_fd),
        ]
        comb_fd = AdaptiveCrossApproximation.CombinedPivStrat(strats_fd)

        conv_fd = AdaptiveCrossApproximation.CombinedConvCritFunctor(
            [
                AdaptiveCrossApproximation.FNormEstimator(1e-4)(),
                AdaptiveCrossApproximation.FNormEstimator(1e-4)(),
            ],
            [false, true],
        )

        idcs_fd = [2, 4, 6]
        functor_fd = comb_fd(conv_fd, idcs_fd)
        chosen_fd = functor_fd(zeros(length(idcs_fd)))

        @test chosen_fd == 1
        @test all(isapprox.(functor_fd.strats[2].h, [0.0, 2.0, 4.0]))

        reset!(functor_fd, [2, 4, 6, 1, 5, 6, 2])
        @test length(functor_fd.strats[1].usedidcs) == 7
        @test length(functor_fd.strats[2].h) == 7
        @test functor_fd.strats[2].nactive == 7

        reset!(functor_fd, [2, 4, 6])
        chosen_fd_reset = functor_fd(zeros(3))
        @test chosen_fd_reset == 1
        @test length(functor_fd.strats[2].h) == 7
        @test functor_fd.strats[2].nactive == 3
        @test all(isapprox.(view(functor_fd.strats[2].h, 1:3), [0.0, 2.0, 4.0]))
    end
end
