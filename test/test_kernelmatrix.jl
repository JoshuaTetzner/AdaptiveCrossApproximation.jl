using BEAST
using H2Trees
using CompScienceMeshes
using AdaptiveCrossApproximation

##

Γ = meshsphere(1.0, 0.1)
op = Helmholtz3D.singlelayer()
space = lagrangecxd0(Γ)

##
tree = TwoNTree(space, space, 1 / 2^10; minvaluestest=100, minvaluestrial=100)

permt = AdaptiveCrossApproximation.H.permutation(H2Trees.testtree(tree))
perms = AdaptiveCrossApproximation.H.permutation(H2Trees.trialtree(tree))
permt == perms
AdaptiveCrossApproximation.H.permute!(space, permt)
##

@time nn = AdaptiveCrossApproximation.H.nearinteractions(tree);
@time AdaptiveCrossApproximation.H.farinteractions(tree);

##
ass = AdaptiveCrossApproximation.H.AbstractKernelMatrix(op, space, space)
##

mat = AdaptiveCrossApproximation.H.assemblenears(op, space, space, tree;);

##

AdaptiveCrossApproximation.H.assemblefars(op, space, space, tree;)
