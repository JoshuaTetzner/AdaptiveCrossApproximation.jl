module ACAH2Trees
using H2Trees
import H2Trees: isleaf, testtree, trialtree, root, children, parent
using StaticArrays
using LinearAlgebra
using AdaptiveCrossApproximation
import AdaptiveCrossApproximation: GeoPivStrat, GeoPivStratFunctor

include("treemimicrypivoting.jl")

function reorder(tree::H2Trees.H2ClusterTree)
    permutation = zeros(Int, H2Trees.numberofvalues(tree))
    n = 1
    for leaf in H2Trees.leaves(tree)
        permutation[n:(n + length(H2Trees.values(tree, leaf)) - 1)] = H2Trees.values(
            tree, leaf
        )
        tree.nodes[leaf].data.values .= n:(n + length(H2Trees.values(tree, leaf)) - 1)
        n += length(H2Trees.values(tree, leaf))
    end
    return permutation
end

end
