using ACAFact
using AdaptiveCrossApproximation

pts1 = range(0.0, 1.0; length=100)
pts2 = range(0.0, 1.0; length=110)
K = [exp(-abs2(pj - pk)) for pj in pts1, pk in pts2]

# AdaptiveCrossApproximation.jl
aca = ACA(
    AdaptiveCrossApproximation.MaximumValue(zeros(Bool, size(K, 1))),
    AdaptiveCrossApproximation.MaximumValue(zeros(Bool, size(K, 2))),
    AdaptiveCrossApproximation.FNormEstimator(0.0),
)

rowbuffer = zeros(Float64, 50, size(K, 2))
colbuffer = zeros(Float64, size(K, 1), 50)

@time npivots = aca(K, rowbuffer, colbuffer, 50, 1e-4);

# ACAFact.jl
cache = ACAFact.ACACache(Float64, length(pts1), length(pts2), 50) # max rank 50
(U, V) = (zeros(length(pts1), 50), zeros(length(pts2), 50))
@time (rnk, _, _) = ACAFact.aca!(K, U, V, 1e-4; cache=cache); # zero allocations
