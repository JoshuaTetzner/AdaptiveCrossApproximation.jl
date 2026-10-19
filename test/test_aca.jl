using AdaptiveCrossApproximation
using LinearAlgebra
using Random
using StaticArrays
using Test

@testset "aca.jl" begin
    @testset "aca / ACA" begin
        Random.seed!(1234)

        tpos = [@SVector rand(3) for _ in 1:48]
        spos = [@SVector rand(3) for _ in 1:52] .+ Scalar(SVector(3.5, 0.0, 0.0))
        Kc = [inv(norm(tp - sp)) for tp in tpos, sp in spos]

        rowpivotings = [
            AdaptiveCrossApproximation.MaximumValue(),
            AdaptiveCrossApproximation.Leja2(tpos),
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

        U, V = AdaptiveCrossApproximation.aca(Kc; tol=1e-4, maxrank=30, svdrecompress=true)
        @test size(U, 1) == size(Kc, 1)
        @test size(V, 2) == size(Kc, 2)
        @test size(U, 2) == size(V, 1)
        @test norm(U * V - Kc) / norm(Kc) < 2e-4

        rowidcs = collect(1:2:size(Kc, 1))
        colidcs = collect(1:2:size(Kc, 2))
        maxrank = 20
        compressor = AdaptiveCrossApproximation.ACA(; tol=1e-4)(
            Kc, length(rowidcs), length(colidcs), maxrank
        )
        colbuffer = zeros(length(rowidcs), maxrank)
        rowbuffer = zeros(maxrank, length(colidcs))
        npivots = compressor(
            Kc, colbuffer, rowbuffer, maxrank; rowidcs=rowidcs, colidcs=colidcs
        )
        @test norm(
            colbuffer[:, 1:npivots] * rowbuffer[1:npivots, :] - Kc[rowidcs, colidcs]
        ) / norm(Kc[rowidcs, colidcs]) < 2e-4

        reverse!(rowidcs)
        reverse!(colidcs)
        fill!(colbuffer, 0)
        fill!(rowbuffer, 0)
        npivots = compressor(
            Kc, colbuffer, rowbuffer, maxrank; rowidcs=rowidcs, colidcs=colidcs
        )
        @test norm(
            colbuffer[:, 1:npivots] * rowbuffer[1:npivots, :] - Kc[rowidcs, colidcs]
        ) / norm(Kc[rowidcs, colidcs]) < 2e-4
    end

    @testset "aca (zero and exact-rank matrices)" begin
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

    @testset "aca (maxrank)" begin
        Random.seed!(21)
        M = randn(10, 10)  # generically full rank (10), so maxrank must bind
        U, V = AdaptiveCrossApproximation.aca(M; tol=1e-14, maxrank=4)
        @test size(U, 2) == 4
        @test size(V, 1) == 4
        @test norm(U * V - M) / norm(M) > 1e-8  # genuinely capped, not converged
    end

    @testset "ACA (pivot tie-breaking)" begin
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

    @testset "aca (scale invariance)" begin
        Random.seed!(3)
        x = [rand(3) for _ in 1:60]
        y = [rand(3) .+ [3.0, 0.0, 0.0] for _ in 1:70]
        K = [1 / norm(a - b) for a in x, b in y]
        for (T, tol) in ((Float32, 1e-5), (Float64, 1e-5), (Float64, 1e-10))
            ranks = map((1e-16, 1.0, 1e16)) do scale
                M = T.(scale .* K)
                U, V = AdaptiveCrossApproximation.aca(M; tol=tol, maxrank=60)
                @test norm(U * V - M) / norm(M) < 10tol
                return size(U, 2)
            end
            @test allequal(ranks)
        end
    end

    @testset "aca (underflow warning)" begin
        M = Float32.(1e-30 .* (randn(20, 2) * randn(2, 30)))
        @test_logs (:warn, r"underflow") match_mode = :any AdaptiveCrossApproximation.aca(
            M; tol=1e-5, maxrank=10
        )
        M = Float32.(1e-16 .* (randn(20, 2) * randn(2, 30)))
        @test_logs AdaptiveCrossApproximation.aca(M; tol=1e-5, maxrank=10)
    end

    @testset "aca (exact-rank termination)" begin
        Random.seed!(4)
        for T in (Float32, Float64), r in (1, 3, 6), scale in (1e-16, 1.0, 1e16)
            tol = T == Float32 ? 1e-5 : 1e-10
            # Orthonormal factors keep the exact-rank roundoff floor near eps.
            Q1 = Matrix(qr(randn(50, r)).Q)
            Q2 = Matrix(qr(randn(40, r)).Q)
            M = T.(scale .* (Q1 * Diagonal([0.9^k for k in 0:(r - 1)]) * Q2'))
            U, V = AdaptiveCrossApproximation.aca(M; tol=tol, maxrank=20)
            @test r <= size(U, 2) <= r + 1
            @test norm(U * V - M) / norm(M) < tol
        end
    end
end
