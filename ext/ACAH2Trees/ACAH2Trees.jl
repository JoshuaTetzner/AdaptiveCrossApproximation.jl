module ACAH2Trees
using H2Trees
import H2Trees:
    isleaf, testtree, trialtree, root, children, firstchild, data, numberofvalues
using LinearAlgebra
using AdaptiveCrossApproximation

include("treemimicrypivoting.jl")
function AdaptiveCrossApproximation._tree(
    ::AdaptiveCrossApproximation.H2Tree,
    testspace,
    trialspace;
    minhalfsize=1 / 2^10,
    testminvalues::Int=200,
    trialminvalues::Int=200,
    kwargs...,
)
    return H2Trees.TwoNTree(
        testspace,
        trialspace,
        minhalfsize;
        testminvalues=testminvalues,
        trialminvalues=trialminvalues,
        kwargs...,
    )
end

function AdaptiveCrossApproximation.permutation(tree::H2Trees.H2ClusterTree)
    perm = zeros(Int, H2Trees.numberofvalues(tree))
    n = 1
    for leaf in H2Trees.leaves(tree)
        perm[n:(n + length(H2Trees.values(tree, leaf)) - 1)] = H2Trees.values(tree, leaf)
        tree.nodes[leaf].data.values .= n:(n + length(H2Trees.values(tree, leaf)) - 1)
        n += length(H2Trees.values(tree, leaf))
    end
    return perm
end

function firstvalue(tree::H2Trees.H2ClusterTree, node::Int)
    iszero(firstchild(tree, node)) && (return data(tree, node).values[1])
    return firstvalue(tree, firstchild(tree, node))
end

function range(tree::H2Trees.H2ClusterTree, node::Int)
    return UnitRange(
        firstvalue(tree, node), (firstvalue(tree, node) + numberofvalues(tree, node) .- 1)
    )
end

AdaptiveCrossApproximation.testtree(tree::H2Trees.BlockTree) = testtree(tree)
AdaptiveCrossApproximation.trialtree(tree::H2Trees.BlockTree) = trialtree(tree)
AdaptiveCrossApproximation.values(tree::H2Trees.H2ClusterTree, node::Int) =
    H2Trees.values(tree, node)
AdaptiveCrossApproximation.levels(tree::H2Trees.H2ClusterTree) = H2Trees.levels(tree)
AdaptiveCrossApproximation.LevelIterator(tree::H2Trees.H2ClusterTree, level::Int) =
    H2Trees.LevelIterator(tree, level)

include("nearinteractions.jl")
include("farinteractions.jl")

end
