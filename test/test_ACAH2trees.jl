using ParallelKMeans
using H2Trees
using AdaptiveCrossApproximation
using StaticArrays
using LinearAlgebra
using Test
using Random
Random.seed!(1)

pts1 = [@SVector rand(3) for i in 1:1000]
pts2 = [@SVector rand(3) for i in 1:1100] .+ Scalar(SVector(1.0, 0.0, 0.0))
K = [1 / (norm(pj - pk)) for pj in pts1, pk in pts2]
##
@time ttree = KMeansTree(pts1, 4; minvalues=20);
stree = KMeansTree(pts2, 4; minvalues=20)
tree = BlockTree(ttree, stree)

Ft = collect(H2Trees.WellSeparatedIterator(tree.trialcluster, tree.testcluster, 3))
rowidcs = H2Trees.values(ttree, 3)
colidcs = Int[]
for s in Ft
    append!(colidcs, H2Trees.values(stree, s))
end
maxrank = min(length(rowidcs), 40)

##

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
@time npiv, rows, cols = iaca(
    K, colbuffer, rowbuffer, maxrank; rows=rows, cols=cols, rowidcs=rowidcs, colidcs=Ft
)
println(npiv)
norm(K[rowidcs, colidcs] - K[rowidcs, cols] * K[rows, cols]^-1 * K[rows, colidcs]) /
norm(K[rowidcs, colidcs])

##
iaca = AdaptiveCrossApproximation.iACA(
    AdaptiveCrossApproximation.MaximumValue(),
    AdaptiveCrossApproximation.MimicryPivoting(pts1, pts2),
    AdaptiveCrossApproximation.FNormExtrapolator(
        AdaptiveCrossApproximation.iFNormEstimator(1e-3)
    ),
)

colbuffer = zeros(Float64, length(rowidcs), maxrank)
rowbuffer = zeros(Float64, maxrank, maxrank)
rows = zeros(Int, maxrank)
cols = zeros(Int, maxrank)
npiv, rows, cols = iaca(
    K, colbuffer, rowbuffer, maxrank; rows=rows, cols=cols, rowidcs=rowidcs, colidcs=colidcs
)
println(npiv)
norm(K[rowidcs, colidcs] - K[rowidcs, cols] * K[rows, cols]^-1 * K[rows, colidcs]) /
norm(K[rowidcs, colidcs])
##
norm(K[:, cols] * inv(K[rows, cols]) * K[rows, :] - K) / norm(K)
##
iaca = AdaptiveCrossApproximation.iACA(
    AdaptiveCrossApproximation.MaximumValue(),
    AdaptiveCrossApproximation.TreeMimicryPivoting(pts2, pts1, ttree),
    AdaptiveCrossApproximation.FNormExtrapolator(
        AdaptiveCrossApproximation.iFNormEstimator(1e-3)
    ),
)
iaca = iaca(colidcs, Fs, maxrank)
rowbuffer = zeros(Float64, maxrank, maxrank)
colbuffer = zeros(Float64, length(colidcs), maxrank)
rows = zeros(Int, maxrank)
cols = zeros(Int, maxrank)
@time npiv, rows, cols = iaca(
    transpose(K), colbuffer, rowbuffer, maxrank, rows, cols, colidcs
)

@test norm(K[rowidcs, colidcs] - K[rowidcs, rows] * K[cols, rows]^-1 * K[cols, colidcs]) /
      norm(K[rowidcs, colidcs]) < 1e-3
