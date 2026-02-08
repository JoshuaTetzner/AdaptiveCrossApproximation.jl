using H2Trees
import H2Trees: testtree, trialtree
using Test
using BEAST
using CompScienceMeshes
using StaticArrays
using AdaptiveCrossApproximation

Γ = meshrectangle(1.0, 1.0, 0.05)
points = Γ.vertices

tree = TwoNTree(points, points, 1 / 2^10; minvaluestest=100, minvaluestrial=100)

permutationtest = AdaptiveCrossApproximation.permutation(H2Trees.testtree(tree))
permutationtrial = AdaptiveCrossApproximation.permutation(H2Trees.trialtree(tree))
@assert permutationtest == permutationtrial
permute!(points, permutationtest)

@time nvalues, nearvalues = AdaptiveCrossApproximation.nearinteractions(tree);
@time fvalptr, fvalues, farvalues = AdaptiveCrossApproximation.farinteractions(tree);

A = zeros(Bool, length(points), length(points))

for (val, near) in zip(nvalues, nearvalues)
    @test !any(A[val, collect(Iterators.flatten(near))])
    A[val, collect(Iterators.flatten(near))] .= true
end

for idx in eachindex(fvalptr[1:(end - 1)])
    for farvalue in farvalues[fvalptr[idx]:(fvalptr[idx + 1] - 1)]
        @test !any(A[fvalues[idx], farvalue])
        A[fvalues[idx], farvalue] .= true
    end
end
@test all(A)

##

Γ = meshsphere(1.0, 0.1)

op = Helmholtz3D.singlelayer()
space = lagrangecxd0(Γ)

tree = TwoNTree(space, space, 1 / 2^10; minvaluestest=100, minvaluestrial=100)

permutationtest = AdaptiveCrossApproximation.permutation(H2Trees.testtree(tree))
permutationtrial = AdaptiveCrossApproximation.permutation(H2Trees.trialtree(tree))
@assert permutationtest == permutationtrial
AdaptiveCrossApproximation.permute!(space, permutationtest)
##
@time nvalues, nearvalues = AdaptiveCrossApproximation.nearinteractions(tree);
@time fvalptr, fvalues, farvalues = AdaptiveCrossApproximation.farinteractions(tree);
farvalues
##
@time mat = AdaptiveCrossApproximation.assemblenears(op, space, space, tree;);
@time fmat = AdaptiveCrossApproximation.assemblefars(op, space, space, tree;);
##
a = assemble(op, space, space)

x = randn(length(space))
y = fmat[1] * x + mat * x
yt = a * x

using LinearAlgebra
norm(y - yt) / norm(yt)
