using H2Trees
using AdaptiveCrossApproximation
using StaticArrays
using LinearAlgebra
using Test
using Random
Random.seed!(1)

pts1 = [@SVector rand(3) for i in 1:200]
pts2 = [@SVector rand(3) for i in 1:200] .+ Scalar(SVector(1.0, 0.0, 0.0))
K = [1 / (norm(pj - pk)) for pj in pts1, pk in pts2]

##

ttree = TwoNTree(pts1, 0.01; minvalues=100);
stree = TwoNTree(pts2, 0.01; minvalues=100)
tree = BlockTree(ttree, stree)

Ft = collect(H2Trees.WellSeparatedIterator(tree.trialcluster, tree.testcluster, 3))
rowidcs = H2Trees.values(ttree, 3)
colidcs = Int[]
for s in Ft
    append!(colidcs, H2Trees.values(stree, s))
end
maxrank = min(length(rowidcs), 40)

iaca = AdaptiveCrossApproximation.iACA(
    AdaptiveCrossApproximation.MaximumValue(),
    AdaptiveCrossApproximation.TreeMimicryPivoting(pts1, pts2, stree),
    AdaptiveCrossApproximation.FNormExtrapolator(
        AdaptiveCrossApproximation.iFNormEstimator(1e-3)
    ),
)
colbuffer = zeros(Float64, length(rowidcs), maxrank)
rowbuffer = zeros(Float64, maxrank, maxrank)
rows = zeros(Int, maxrank)
cols = zeros(Int, maxrank)
npiv, rows, cols = iaca(
    K, colbuffer, rowbuffer, maxrank; rows=rows, cols=cols, rowidcs=rowidcs, colidcs=Ft
)
@test norm(K[rowidcs, colidcs] - K[rowidcs, cols] * K[rows, cols]^-1 * K[rows, colidcs]) /
      norm(K[rowidcs, colidcs]) < 1e-3

Fs = collect(H2Trees.WellSeparatedIterator(tree.testcluster, tree.trialcluster, 3))
colidcs = H2Trees.values(stree, 3)
rowidcs = Int[]
for s in Fs
    append!(rowidcs, H2Trees.values(ttree, s))
end

iaca = AdaptiveCrossApproximation.iACA(
    AdaptiveCrossApproximation.TreeMimicryPivoting(pts2, pts1, ttree),
    AdaptiveCrossApproximation.MaximumValue(),
    AdaptiveCrossApproximation.FNormExtrapolator(
        AdaptiveCrossApproximation.iFNormEstimator(1e-3)
    ),
)
colbuffer = zeros(Float64, maxrank, maxrank)
rowbuffer = zeros(Float64, maxrank, length(colidcs))
rows = zeros(Int, maxrank)
cols = zeros(Int, maxrank)
npiv, rows, cols = iaca(
    K, colbuffer, rowbuffer, maxrank; rows=rows, cols=cols, rowidcs=Fs, colidcs=colidcs
)
@test norm(K[rowidcs, colidcs] - K[rowidcs, cols] * K[rows, cols]^-1 * K[rows, colidcs]) /
      norm(K[rowidcs, colidcs]) < 1e-3
