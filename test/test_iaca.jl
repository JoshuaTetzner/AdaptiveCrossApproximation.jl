using AdaptiveCrossApproximation
using H2Trees
using LinearAlgebra
using Random
using StaticArrays
using Test

plate = meshrectangle(1.0, 1.0, 0.1)

Xs = raviartthomas(plate)
Xt = raviartthomas(translate(plate, [0.0, 0.0, 3.0]))

tree = TwoNTree(Xs.pos, 0.0; minvalues=40)

op = Maxwell3D.singlelayer(; wavenumber=k)
A = assemble(op, Xt, Xs)
tol = 1e-2
maxrank = 40
iaca = iACA(
    MaximumValue(),
    TreeMimicryPivoting(Xt.pos, Xs.pos, tree),
    FNormExtrapolator(iFNormEstimator(tol)),
)
iaca = iaca([1], [1], maxrank)
iaca2 = iACA(
    MaximumValue(), MimicryPivoting(Xt.pos, Xs.pos), FNormExtrapolator(iFNormEstimator(tol))
)
iaca2 = iaca2([1], [1], maxrank)

##
rowbuffer = zeros(eltype(A), maxrank, maxrank)
colbuffer = zeros(eltype(A), size(A, 1), maxrank)
rowidcs = Vector(1:size(A, 1))
colidcs = collect(H2Trees.LevelIterator(tree, 2))
rowpivs = zeros(Int, maxrank)
colpivs = zeros(Int, maxrank)

npivots, rows, cols = iaca(
    A, colbuffer, rowbuffer, rowpivs, colpivs, rowidcs, colidcs, maxrank
)
norm(A[:, cols] * inv(A[rows, cols]) * A[rows, :] - A) / norm(A)
##
rowbuffer = zeros(eltype(A), maxrank, maxrank)
colbuffer = zeros(eltype(A), size(A, 1), maxrank)
rowidcs = Vector(1:size(A, 1))
colidcs = Vector(1:size(A, 2))
rowpivs = zeros(Int, maxrank)
colpivs = zeros(Int, maxrank)

npivots, rows, cols = iaca2(
    A, colbuffer, rowbuffer, rowpivs, colpivs, rowidcs, colidcs, maxrank
)
norm(A[:, cols] * inv(A[rows, cols]) * A[rows, :] - A) / norm(A)
##
