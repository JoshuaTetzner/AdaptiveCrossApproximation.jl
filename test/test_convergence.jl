using AdaptiveCrossApproximation
using LinearAlgebra
using Random
using Test

pts1 = range(0.0, 1.0; length=100)
pts2 = range(0.0, 1.0; length=110)
K = [exp(-abs2(pj - pk)) for pj in pts1, pk in pts2]

# FNormExtrapolator
for tol in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14]
    local U, V = AdaptiveCrossApproximation.aca(
        K; tol=tol, convergence=AdaptiveCrossApproximation.FNormExtrapolator(Float64)
    )
    @test norm(U * V - K) / norm(K) < tol
end

# RandomSampling
for tol in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14]
    Random.seed!(1)
    indices = hcat(rand(1:100, 100), rand(1:110, 100))
    rest = [K[rc[1], rc[2]] for rc in eachrow(indices)]
    convergence = AdaptiveCrossApproximation.RandomSampling(0.0, 100, 1.0, indices, rest)
    local U, V = AdaptiveCrossApproximation.aca(K; tol=tol, convergence=convergence)
    @test norm(U * V - K) / norm(K) < tol
end

# CombinedConvCrit
for tol in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14]
    Random.seed!(1)
    cc1 = AdaptiveCrossApproximation.FNormEstimator(Float64)
    indices = hcat(rand(1:100, 100), rand(1:110, 100))
    rest = [K[rc[1], rc[2]] for rc in eachrow(indices)]
    cc2 = AdaptiveCrossApproximation.RandomSampling(0.0, 100, 1.0, indices, rest)

    convergence = AdaptiveCrossApproximation.CombinedConvCrit([cc1, cc2], zeros(Bool, 2))

    local U, V = AdaptiveCrossApproximation.aca(K; tol=tol, convergence=convergence)

    @test norm(U * V - K) / norm(K) < tol
end
