using AdaptiveCrossApproximation
using LinearAlgebra
using Test

pts1 = range(0.0, 1.0; length=100)
pts2 = range(0.0, 1.0; length=110)
K = [exp(-abs2(pj - pk)) for pj in pts1, pk in pts2]

# accuracy validation

for tol in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14]
    local U, V = AdaptiveCrossApproximation.aca(K; tol=tol)
    norm(U * V - K) / norm(K)
    @test norm(U * V - K) / norm(K) < tol
end

# special cases

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
