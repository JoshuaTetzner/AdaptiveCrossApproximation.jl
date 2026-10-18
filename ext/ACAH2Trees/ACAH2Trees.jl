module ACAH2Trees
using H2Trees
import H2Trees:
    isleaf, testtree, trialtree, root, children, firstchild, data, numberofvalues
using LinearAlgebra
using AdaptiveCrossApproximation
using AdaptiveCrossApproximation:
    IsNearFunctor, SpaceOrderingStyle, PreserveSpaceOrder, PermuteSpaceInPlace

include("treemimicrypivoting.jl")
function AdaptiveCrossApproximation._tree(
    ::AdaptiveCrossApproximation.H2Tree,
    space;
    minhalfsize=1 / 2^10,
    minvalues::Int=200,
    testminvalues::Int=minvalues,
    trialminvalues::Int=testminvalues,
    kwargs...,
)
    # Accept the existing two-space keywords, but only construct the test tree.
    cluster = if isdefined(H2Trees, :TwoNTreeBuilder)
        H2Trees.buildtree(
            space;
            builder=H2Trees.TwoNTreeBuilder(;
                minhalfsize=minhalfsize, minvalues=testminvalues, kwargs...
            ),
        )
    else
        H2Trees.TwoNTree(
            AdaptiveCrossApproximation._treepositions(space),
            minhalfsize;
            minvalues=testminvalues,
            kwargs...,
        )
    end
    return H2Trees.BlockTree(cluster, cluster)
end

function AdaptiveCrossApproximation._tree(
    ::AdaptiveCrossApproximation.H2Tree,
    testspace,
    trialspace;
    minhalfsize=1 / 2^10,
    testminvalues::Int=200,
    trialminvalues::Int=200,
    kwargs...,
)
    if isdefined(H2Trees, :BlockTreeBuilder)
        return H2Trees.buildtree(
            testspace,
            trialspace;
            builder=H2Trees.BlockTreeBuilder(;
                test=H2Trees.TwoNTreeBuilder(;
                    minhalfsize=minhalfsize, minvalues=testminvalues, kwargs...
                ),
                trial=H2Trees.TwoNTreeBuilder(;
                    minhalfsize=minhalfsize, minvalues=trialminvalues, kwargs...
                ),
            ),
        )
    end
    return H2Trees.TwoNTree(
        AdaptiveCrossApproximation._treepositions(testspace),
        AdaptiveCrossApproximation._treepositions(trialspace),
        minhalfsize;
        testminvalues=testminvalues,
        trialminvalues=trialminvalues,
        kwargs...,
    )
end

function AdaptiveCrossApproximation.permutation(tree::H2Trees.H2ClusterTree)
    return collect(H2Trees.values(tree, H2Trees.root(tree)))
end

function AdaptiveCrossApproximation._settreeorder!(tree::H2Trees.H2ClusterTree)
    n = 1
    for leaf in H2Trees.leaves(tree)
        leafvalues = H2Trees.values(tree, leaf)
        leafvalues .= n:(n + length(leafvalues) - 1)
        n += length(leafvalues)
    end
    return tree
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

_interactionindices(tree, node, ::PreserveSpaceOrder) = H2Trees.values(tree, node)
_interactionindices(tree, node, ::PermuteSpaceInPlace) = collect(range(tree, node))

AdaptiveCrossApproximation.testtree(tree::H2Trees.BlockTree) = testtree(tree)
AdaptiveCrossApproximation.trialtree(tree::H2Trees.BlockTree) = trialtree(tree)
AdaptiveCrossApproximation.values(tree::H2Trees.H2ClusterTree, node::Int) =
    H2Trees.values(tree, node)
AdaptiveCrossApproximation.levels(tree::H2Trees.H2ClusterTree) = H2Trees.levels(tree)
AdaptiveCrossApproximation.LevelIterator(tree::H2Trees.H2ClusterTree, level::Int) =
    H2Trees.LevelIterator(tree, level)

include("terminalinteractions.jl")
include("nearinteractions.jl")
include("symmetricnearinteractions.jl")
include("farinteractions.jl")
include("symmetricfarinteractions.jl")

end
