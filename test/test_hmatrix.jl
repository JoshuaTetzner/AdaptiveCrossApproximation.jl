using BEAST
using CompScienceMeshes
using H2Trees

Γ = meshsphere(1.0, 0.1)
space = raviartthomas(Γ)
tree = TwoNTree(space, 0.1)

permutation = zeros(Int, length(space))

n = 1

for leaf in H2Trees.leaves(tree)
    permutation[n:(n + length(H2Trees.values(tree, leaf)) - 1)] = H2Trees.values(tree, leaf)
    tree.nodes[leaf].data.values .= n:(n + length(H2Trees.values(tree, leaf)) - 1)
    n += length(H2Trees.values(tree, leaf))
end
permute!(space.fns, permutation)
permute!(space.pos, permutation)
##

using BlockSparseMatrices
using OhMyThreads

#=
struct VariableBlockCompressedRowStorage{T,M,P<:Integer,S} <: AbstractBlockMatrix{T}
    blocks::Vector{M}
    rowptr::Vector{P}
    colindices::Vector{P}
    rowindices::Vector{P}
    size::Tuple{Int,Int}
    scheduler::S
end
=#

a = [rand(10, 10) for i in 1:10]
rows = [1, 11, 21]
cols = [1, 11, 21, 31, 1, 11, 21, 21, 31, 41]
dim = (30, 50)
rowptr = [1, 5, 8, 11]
sc = SerialScheduler()
typeof(sc)
mat = VariableBlockCompressedRowStorage{Float64,Matrix{Float64},Int,SerialScheduler}(
    a, rowptr, cols, rows, dim, SerialScheduler()
)

x = rand(50)
mat * x
##

a = [rand(10, 10) for i in 1:4]
rows = [Vector(1:10), Vector(1:10), Vector(11:20), Vector(11:20)]
cols = [Vector(1:10), Vector(11:20), Vector(1:10), Vector(11:20)]

B = BlockSparseMatrices.BlockSparseMatrix(a, cols, rows, (20, 20))

V = VariableBlockCompressedRowStorage(B)
V.rowptr
V.rowindices
V.colindices
