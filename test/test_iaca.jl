using AdaptiveCrossApproximation
using LinearAlgebra
using StaticArrays
using Random
using Test
##
Random.seed!(1)
pts1 = [@SVector rand(3) for i in 1:100]
pts2 = [@SVector rand(3) for i in 1:110] .+ Scalar(SVector(4.0, 0.0, 0.0))
K = [1 / (norm(pj - pk)) for pj in pts1, pk in pts2]

iaca = AdaptiveCrossApproximation.iACA(
    AdaptiveCrossApproximation.MimicryPivoting(pts2, pts1),
    AdaptiveCrossApproximation.MaximumValue(),
    AdaptiveCrossApproximation.FNormExtrapolator(
        AdaptiveCrossApproximation.iFNormEstimator(1e-4)
    ),
)
iaca(Vector(1:100), Vector(1:110))
##
colbuffer = zeros(Float64, 40, 40)
rowbuffer = zeros(Float64, 40, 110)
npiv, rows, cols = iaca(K, colbuffer, rowbuffer, 40)
rows

##
norm(K[:, cols] * inv(K[rows, cols]) * K[rows, :] - K) / norm(K)

##
iaca = AdaptiveCrossApproximation.iACA(
    AdaptiveCrossApproximation.MaximumValue(),
    AdaptiveCrossApproximation.MimicryPivoting(pts2, pts1),
    AdaptiveCrossApproximation.FNormExtrapolator(
        AdaptiveCrossApproximation.iFNormEstimator(1e-4)
    ),
)
iaca(Vector(1:110), Vector(1:100))

colbuffer = zeros(Float64, 110, 40)
rowbuffer = zeros(Float64, 40, 40)
npiv, rows, cols = iaca(Matrix(transpose(K)), colbuffer, rowbuffer, 40)

rows
##
K = transpose(K)
norm(K[:, cols] * inv(K[rows, cols]) * K[rows, :] - K) / norm(K)

##
pts1 = [@SVector rand(3) for i in 1:100]
pts2 = [@SVector rand(3) for i in 1:100] .+ Scalar(SVector(4.0, 0.0, 0.0))
##
K = [1 / (norm(pj - pk)) for pj in pts1, pk in pts2]
Kt = Matrix(transpose(K))

iaca1 = AdaptiveCrossApproximation.iACA(
    AdaptiveCrossApproximation.MimicryPivoting(pts2, pts1),
    AdaptiveCrossApproximation.MaximumValue(),
    AdaptiveCrossApproximation.FNormExtrapolator(
        AdaptiveCrossApproximation.iFNormEstimator(1e-4)
    ),
)
colbuffer = zeros(Float64, 40, 40)
rowbuffer = zeros(Float64, 40, 100)
npiv, rows, cols = iaca1(K, colbuffer, rowbuffer, 40)

##

iaca2 = AdaptiveCrossApproximation.iACA(
    AdaptiveCrossApproximation.MaximumValue(),
    AdaptiveCrossApproximation.MimicryPivoting(pts2, pts1),
    AdaptiveCrossApproximation.FNormExtrapolator(
        AdaptiveCrossApproximation.iFNormEstimator(1e-4)
    ),
)
colbuffer = zeros(Float64, 100, 40)
rowbuffer = zeros(Float64, 40, 40)
npiv, rows, cols = iaca2(Kt, colbuffer, rowbuffer, 40)
