module ACAH2Trees
using H2Trees
import H2Trees: isleaf, testtree, trialtree, root, children
using LinearAlgebra
using AdaptiveCrossApproximation
function AdaptiveCrossApproximation.buildtree(
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

function AdaptiveCrossApproximation.buildtree(
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

include("clusters.jl")
include("interactions.jl")
include("symmetricinteractions.jl")

end
