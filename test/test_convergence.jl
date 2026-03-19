using AdaptiveCrossApproximation
using LinearAlgebra
using Random
using Test

@testset "FNormEstimatorFunctor" begin
    cc = AdaptiveCrossApproximation.FNormEstimator(0.5)
    functor = cc()

    rowbuffer = [1.0 0.0 0.0; 0.0 2.0 0.0]
    colbuffer = [3.0 0.0; 0.0 4.0]

    npivot1, conv1 = functor(rowbuffer, colbuffer, 1, 2, 3)
    @test npivot1 == 1
    @test conv1
    @test isapprox(functor.normUV², 9.0)
    @test AdaptiveCrossApproximation.tolerance(cc) == 0.5

    npivot2, conv2 = functor(rowbuffer, colbuffer, 2, 2, 3)
    @test npivot2 == 2
    @test conv2
    @test isapprox(functor.normUV², 73.0)

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
    @test iszero(functor.normUV²)
    @test resize!(functor, 10) === nothing
end

@testset "iFNormEstimatorFunctor" begin
    cc = AdaptiveCrossApproximation.iFNormEstimator(0.8)
    functor = cc()

    buf1 = [3.0, 4.0]
    AdaptiveCrossApproximation.normF!(functor, buf1, 1)
    @test isapprox(functor.normUV, 5.0)
    @test AdaptiveCrossApproximation.tolerance(functor) == 0.8

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
    @test resize!(functor, 12) === nothing
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
    @test iszero(functor.estimator.normUV²)
    @test resize!(functor, 12) === nothing

    cc_iaca = AdaptiveCrossApproximation.FNormExtrapolator(
        AdaptiveCrossApproximation.iFNormEstimator(0.5)
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
    rhs = functor.tol * (rnorm * cnorm)

    npivot, conv = functor(rowbuffer, colbuffer, 1, 3, 4)
    @test npivot == 1
    @test conv == (lhs > rhs)
    @test all(isapprox.(functor.rest, expected_after))
    @test isapprox(functor.normUV², (rnorm * cnorm)^2)

    reset!(functor)
    @test iszero(functor.normUV²)
    @test all(iszero, view(functor.rest, 1:(functor.nactive)))
    @test all(==((0, 0)), view(functor.indices, 1:(functor.nactive)))

    rowidcs2 = [1, 3]
    colidcs2 = [2, 4]
    Random.seed!(22)
    resize!(functor, rowidcs2, colidcs2)
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
    @test all(functor.isconverged)

    rowbuffer = zeros(1, 3)
    colbuffer = zeros(3, 1)
    rowbuffer[1, :] .= [1.0, 0.0, 0.0]
    colbuffer[:, 1] .= [1.0, 1.0, 0.0]

    npivot, conv = functor(rowbuffer, colbuffer, 1, 3, 3)
    @test npivot == 1
    @test conv == any(functor.isconverged)

    reset!(functor)
    @test all(functor.isconverged)
    @test iszero(functor.crits[1].normUV²)
    @test iszero(functor.crits[2].normUV²)
    @test all(iszero, view(functor.crits[2].rest, 1:(functor.crits[2].nactive)))
    @test all(==((0, 0)), view(functor.crits[2].indices, 1:(functor.crits[2].nactive)))

    resize!(functor, [1, 2], [1, 2, 3])
    @test all(functor.isconverged)
    @test functor.crits[2].nactive == 3
    @test length(functor.crits[2].indices) >= 3
    @test length(functor.crits[2].rest) >= 3
end
