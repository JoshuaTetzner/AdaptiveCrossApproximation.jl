using AdaptiveCrossApproximation
using LinearAlgebra
using Random
using StaticArrays
using Test

@testset "ACAᵀ" begin
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
                U, V = AdaptiveCrossApproximation.acaᵀ(
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

@testset "ACAᵀ Special Cases" begin
    Random.seed!(1234)
    K = zeros(10, 10)
    U, V = AdaptiveCrossApproximation.acaᵀ(K; tol=10^-4, maxrank=5)
    @test length(U) == 0
    @test length(V) == 0

    K[4, :] = rand(10)
    U, V = AdaptiveCrossApproximation.acaᵀ(K; tol=10^-4, maxrank=5)
    @test size(U, 2) == 1
    @test size(V, 1) == 1

    K[1:2, :] = rand(2, 10)
    U, V = AdaptiveCrossApproximation.acaᵀ(K; tol=10^-4, maxrank=5)
    @test size(U, 2) == 3
    @test size(V, 1) == 3
end

@testset "ACAᵀ is the transpose dual of ACA (MaximumValue)" begin
    # ACAᵀ starts by picking a column first, then a row, alternating from
    # there -- the exact dual of ACA, which starts with a row. With
    # MaximumValue pivoting (deterministic, always starts at index 1), running
    # ACA on M and ACAᵀ on Mᵀ must therefore pick transposed pivots and
    # produce bit-for-bit transposed factors: U*V (from ACA on M) must equal
    # (U'*V')ᵀ (from ACAᵀ on Mᵀ).
    Random.seed!(42)

    for (m, n) in ((30, 22), (22, 30), (17, 17))
        tpos = [@SVector rand(3) for _ in 1:m] .+ Scalar(SVector(5.0, 0.0, 0.0))
        spos = [@SVector rand(3) for _ in 1:n]
        M = [inv(norm(tp - sp)) for tp in tpos, sp in spos]

        U, V = AdaptiveCrossApproximation.aca(M; tol=1e-10, maxrank=min(m, n))
        Uᵀ, Vᵀ = AdaptiveCrossApproximation.acaᵀ(Matrix(M'); tol=1e-10, maxrank=min(m, n))

        @test size(U) == size(Vᵀ')
        @test size(V) == size(Uᵀ')
        @test U == Vᵀ'
        @test V == Uᵀ'
        # U*V and (Uᵀ*Vᵀ)ᵀ are each recomputed via independent BLAS gemm calls
        # (different transpose flags sum the rank dimension in a different
        # order), so only agree up to a few ULP rather than bit-for-bit.
        @test U * V ≈ (Uᵀ * Vᵀ)' rtol = 1e-12
    end
end
