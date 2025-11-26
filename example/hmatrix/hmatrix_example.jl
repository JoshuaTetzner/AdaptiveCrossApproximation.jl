using AdaptiveCrossApproximation
using BEAST
using BlockSparseMatrices
using CompScienceMeshes
using ParallelKMeans
using H2Trees
using LinearAlgebra
using LinearMaps
using OhMyThreads

include("abstractbeastmatrix.jl")
include("skeletons.jl")
include("hmatrix.jl")

##

λ = 2.0
k = 2π / λ
Γ = meshsphere(1.0, 0.1)

op = Maxwell3D.singlelayer(; wavenumber=k)
X = raviartthomas(Γ)

ttree = H2Trees.KMeansTree(X.pos, 2; minvalues=50)
tree = BlockTree(ttree, ttree)

hmat = HMatrix(op, X, X, tree)

##
x = rand(ComplexF64, length(X))

# only the farinteractions are multithreaded, therefore slower than full MV,
# might change for larger problems
@time hmat * x;

A = assemble(op, X, X)
@time A * x;

norm(hmat * x - A * x) / norm(A * x)
