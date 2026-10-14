using AdaptiveCrossApproximation
using LinearAlgebra
using Random
using Test

@testset "Roundoff cutoff preserves small residuals" begin
    for T in (Float32, Float64), F in (Float32, Float64), scale in (1e-16, 1.0, 1e16)
        tol = T == Float32 ? 1e-5 : 1e-12
        relative = T == Float32 ? 5e-5 : 5e-10
        for extrapolate in (false, true)
            criterion = FNormEstimator(F(tol))
            functor = extrapolate ? FNormExtrapolator(criterion)(4) : criterion()
            estimator = extrapolate ? functor.estimator : functor
            estimator.normUV = F(scale)
            # Meaningful residuals below sqrt(eps) must still count as pivots.
            residual = Complex{T}[T(scale*relative)]
            npivot, keepgoing = functor(residual, 2)
            @test npivot == 2
            @test keepgoing
            # A few ulps relative to the block norm remain numerical zero.
            roundoff = Complex{T}[T(scale) * T(4eps(T))]
            npivot, keepgoing = functor(roundoff, 2)
            @test npivot == 1
            @test !keepgoing
        end
    end
end

@testset "Roundoff cutoff is independent of the matrix scale" begin
    for T in (Float32, Float64), scale in (1e-16, 1.0, 1e16)
        tol = T == Float32 ? 1e-5 : 1e-12
        relative = T == Float32 ? 5e-5 : 5e-10
        functor = FNormEstimator(tol)()
        functor.normUV = scale
        # ACA scales the row by its pivot; the column carries the matrix scale.
        rowbuffer = zeros(T, 2, 3)
        colbuffer = zeros(T, 4, 2)
        rowbuffer[2, :] .= (1, 0.5, 0)
        colbuffer[:, 2] .= T(scale * relative)
        npivot, keepgoing = functor(rowbuffer, colbuffer, 2, 4, 3)
        @test npivot == 2
        @test keepgoing
        functor.normUV = scale
        colbuffer[:, 2] .= T(scale) * T(2eps(T))
        npivot, keepgoing = functor(rowbuffer, colbuffer, 2, 4, 3)
        @test npivot == 1
        @test !keepgoing
    end
end

@testset "ACA rank is independent of the matrix scale" begin
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

@testset "ACA warns when squared norms underflow" begin
    M = Float32.(1e-30 .* (randn(20, 2) * randn(2, 30)))
    @test_logs (:warn, r"underflow") match_mode = :any AdaptiveCrossApproximation.aca(
        M; tol=1e-5, maxrank=10
    )
    M = Float32.(1e-16 .* (randn(20, 2) * randn(2, 30)))
    @test_logs AdaptiveCrossApproximation.aca(M; tol=1e-5, maxrank=10)
end

@testset "ACA stops after an exact rank" begin
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

@testset "RandomSamplingFunctor" begin
    K = reshape(collect(1.0:12.0), 3, 4)
    rowidcs = [1, 2, 3]
    colidcs = [1, 2, 3, 4]

    Random.seed!(11)
    cc = AdaptiveCrossApproximation.RandomSampling(; nsamples=5, tol=0.2)
    functor = cc(K, rowidcs, colidcs)

    @test length(functor.indices) == 5
    @test length(unique(functor.indices)) == 5
    @test all(
        1 <= rc[1] <= length(rowidcs) && 1 <= rc[2] <= length(colidcs) for
        rc in functor.indices
    )

    expected_rest = [K[rowidcs[rc[1]], colidcs[rc[2]]] for rc in functor.indices]
    @test all(isapprox.(functor.rest, expected_rest))

    rowbuffer = zeros(2, 4)
    colbuffer = zeros(3, 2)
    rowbuffer[1, :] .= [1.0, 2.0, 0.0, 0.0]
    colbuffer[:, 1] .= [1.0, 0.0, 1.0]

    rest_before = copy(functor.rest)
    expected_after = similar(rest_before)
    for i in eachindex(rest_before)
        rc = functor.indices[i]
        expected_after[i] = rest_before[i] - colbuffer[rc[1], 1] * rowbuffer[1, rc[2]]
    end

    rnorm = norm(rowbuffer[1, 1:4])
    cnorm = norm(colbuffer[1:3, 1])
    lhs = sqrt(sum(abs2, expected_after) / length(expected_after) * 3 * 4)
    rhs = functor.convergence.tol * (rnorm * cnorm)

    npivot, conv = functor(rowbuffer, colbuffer, 1, 3, 4)
    @test npivot == 1
    @test conv == (lhs > rhs)
    @test all(isapprox.(functor.rest, expected_after))
    @test isapprox(functor.normUV, rnorm * cnorm)

    rowidcs2 = [1, 3]
    colidcs2 = [2, 4]
    Random.seed!(22)
    reset!(functor, rowidcs2, colidcs2)
    @test iszero(functor.normUV)
    @test length(functor.indices) >= 4
    @test length(functor.rest) >= 4
    @test functor.nactive == 4
    @test length(unique(view(functor.indices, 1:(functor.nactive)))) == 4
    @test all(
        1 <= rc[1] <= length(rowidcs2) && 1 <= rc[2] <= length(colidcs2) for
        rc in view(functor.indices, 1:(functor.nactive))
    )
    expected_rest2 = [
        K[rowidcs2[rc[1]], colidcs2[rc[2]]] for
        rc in view(functor.indices, 1:(functor.nactive))
    ]
    @test all(isapprox.(view(functor.rest, 1:(functor.nactive)), expected_rest2))
end

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
