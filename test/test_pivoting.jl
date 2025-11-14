using AdaptiveCrossApproximation
using LinearAlgebra
using StaticArrays
using Random
using Test

Random.seed!(3)
pts1 = [@SVector rand(3) for i in 1:100]
pts2 = [@SVector rand(3) for i in 1:110] .+ Scalar(SVector(4.0, 0.0, 0.0))
K = [1 / (norm(pj - pk)) for pj in pts1, pk in pts2]

# leja2
for tol in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14]
    rp = AdaptiveCrossApproximation.Leja2(pts1)
    cp = AdaptiveCrossApproximation.Leja2(pts2)

    local U, V = AdaptiveCrossApproximation.aca(K; rowpivoting=rp, tol=tol, maxrank=100)
    # geometrical pivoting is slightly less accurate
    @test norm(U * V - K) / norm(K) < 2tol

    local U, V = AdaptiveCrossApproximation.aca(K; columnpivoting=cp, tol=tol, maxrank=100)
    # geometrical pivoting is slightly less accurate
    @test norm(U * V - K) / norm(K) < 2tol
end

# fill distance
for tol in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14]
    rp = AdaptiveCrossApproximation.FillDistance(pts1)
    cp = AdaptiveCrossApproximation.FillDistance(pts2)
    local U, V = AdaptiveCrossApproximation.aca(K; rowpivoting=rp, tol=tol, maxrank=100)
    # geometrical pivoting is slightly less accurate
    @test norm(U * V - K) / norm(K) < 2tol

    local U, V = AdaptiveCrossApproximation.aca(K; columnpivoting=cp, tol=tol, maxrank=100)
    # geometrical pivoting is slightly less accurate
    @test norm(U * V - K) / norm(K) < 2tol
end

# combinedpivstrat
for tol in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]#, 1e-12, 1e-14]
    Random.seed!(1)
    cc1 = AdaptiveCrossApproximation.FNormEstimator(tol)
    indices = hcat(rand(1:100, 100), rand(1:110, 100))
    rest = [K[rc[1], rc[2]] for rc in eachrow(indices)]
    cc2 = AdaptiveCrossApproximation.RandomSampling(; tol=tol)
    convergence = AdaptiveCrossApproximation.CombinedConvCrit([cc1, cc2])

    ps1 = MaximumValue()
    ps2 = AdaptiveCrossApproximation.RandomSamplingPivoting(1)
    rp = AdaptiveCrossApproximation.CombinedPivStrat([ps1, ps2])
    local U, V = AdaptiveCrossApproximation.aca(
        K; rowpivoting=rp, convergence=convergence, maxrank=100
    )
    @test norm(U * V - K) / norm(K) < 4tol

    ps3 = MaximumValue()
    ps4 = AdaptiveCrossApproximation.RandomSamplingPivoting(2)
    cp = AdaptiveCrossApproximation.CombinedPivStrat([ps3, ps4])
    local U, V = AdaptiveCrossApproximation.aca(
        K; columnpivoting=cp, convergence=convergence, maxrank=100
    )
    @test norm(U * V - K) / norm(K) < 4tol
end
