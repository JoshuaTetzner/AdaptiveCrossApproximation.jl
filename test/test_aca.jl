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

@testset "ACA maxrank cap" begin
    Random.seed!(21)
    M = randn(10, 10)  # generically full rank (10), so maxrank must bind
    U, V = AdaptiveCrossApproximation.aca(M; tol=1e-14, maxrank=4)
    @test size(U, 2) == 4
    @test size(V, 1) == 4
    @test norm(U * V - M) / norm(M) > 1e-8  # genuinely capped, not converged
end

@testset "ACA tie-break determinism (MaximumValue)" begin
    # Columns 3 and 6 tie at |.|=2.0 in the first (unmodified) row scan;
    # MaximumValueFunctor's `>=` scan means the LAST matching index wins.
    M = zeros(6, 8)
    M[1, 3] = 2.0
    M[1, 6] = 2.0
    M[1, 2] = 1.0

    maxrank = 6
    compressor = AdaptiveCrossApproximation.ACA(; tol=1e-12)
    colbuffer = zeros(size(M, 1), maxrank)
    rowbuffer = zeros(maxrank, size(M, 2))
    rows = zeros(Int, maxrank)
    cols = zeros(Int, maxrank)
    compressor(M, colbuffer, rowbuffer, maxrank; rows=rows, cols=cols)

    @test rows[1] == 1
    @test cols[1] == 6
end
