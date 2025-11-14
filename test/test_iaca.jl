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
colbuffer = zeros(Float64, 40, 40)
rowbuffer = zeros(Float64, 40, 110)
npiv, rows, cols = iaca(K, colbuffer, rowbuffer, 40)

@test norm(K[:, cols] * inv(K[rows, cols]) * K[rows, :] - K) / norm(K) < 1e-4

##
iaca = AdaptiveCrossApproximation.iACA(pts2, pts1)
iaca(Vector(1:110), Vector(1:100))

colbuffer = zeros(Float64, 110, 40)
rowbuffer = zeros(Float64, 40, 40)
npivT, rowsT, colsT = iaca(Matrix(transpose(K)), colbuffer, rowbuffer, 40)

K = transpose(K)
@test norm(K[:, colsT] * inv(K[rowsT, colsT]) * K[rowsT, :] - K) / norm(K) < 1e-4

@test npiv == npivT
@test rows == colsT
@test cols == rowsT
