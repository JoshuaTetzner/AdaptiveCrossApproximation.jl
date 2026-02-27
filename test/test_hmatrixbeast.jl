using BEAST
using CompScienceMeshes
using ParallelKMeans
using H2Trees
using AdaptiveCrossApproximation
using Test
##

λ = 10
k = 2 * π / λ

op = Maxwell3D.singlelayer(; wavenumber=k)

plate = meshrectangle(1.0, 1.0, 0.1)
facingplates = weld(plate, translate(plate, [0.0, 0.0, 1.0]))

meshes = [
    (facingplates, translate(facingplates, [2.0, 0.0, 0.0])), (facingplates, facingplates)
]

for mesh in meshes
    Xt = raviartthomas(mesh[1])
    Xs = raviartthomas(mesh[2])

    x = rand(ComplexF64, length(Xs))

    for tol in [1e-3, 1e-5, 1e-7, 1e-9]
        tree = TwoNTree(Xt, Xs, 1 / 2^10; minvaluestest=100, minvaluestrial=100)

        hmat = AdaptiveCrossApproximation.HMatrix(op, Xt, Xs, tree; tol=tol)
        A = assemble(op, Xt, Xs)
        @test norm(A * x - hmat * x) / norm(A * x) < tol

        testtree = KMeansTree(Xt.pos, 2; minvalues=100)
        trialtree = KMeansTree(Xt.pos, 2; minvalues=100)
        tree = BlockTree(testtree, trialtree)

        hmat = AdaptiveCrossApproximation.HMatrix(op, Xt, Xs, tree; tol=tol)
        A = assemble(op, Xt, Xs)
        @test norm(A * x - hmat * x) / norm(A * x) < tol
    end
end
