using BEAST
using CompScienceMeshes
using H2Trees
using ParallelKMeans
using AdaptiveCrossApproximation
using FastBEAST
Γ = meshsphere(1.0, 0.025)
λ = 1.0
k = 2π / λ
op = Maxwell3D.singlelayer(; wavenumber=k)
space = raviartthomas(Γ)
testtree = KMeansTree(space.pos, 2; minvalues=100)
tree = BlockTree(testtree, testtree)
##
@time hmat = AdaptiveCrossApproximation.HMatrix(op, space, space, tree);
@time FBhmat = HM.PetrovGalerkinHMatrix(
    op, space, space, tree; compressor=ACA(; convergence=FNormEstimator(1e-4))
);

##
x = randn(length(space))

@time y = hmat * x;
@time yFB = FBhmat * x;

size(hmat)
