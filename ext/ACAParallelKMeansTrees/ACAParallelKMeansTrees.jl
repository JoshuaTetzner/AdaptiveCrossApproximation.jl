module ACAParallelKMeansTrees
using BEAST
using H2Trees
using ParallelKMeans
using AdaptiveCrossApproximation

"""
    _tree(::KMeansTreeBackend, testspace::BEAST.Space, trialspace::BEAST.Space; kwargs...)

Build a k-means clustered [`H2Trees.BlockTree`](@ref) from BEAST spaces.

Requires `BEAST.jl`, `H2Trees.jl` and `ParallelKMeans.jl` all loaded (this extension's
trigger). Selected automatically by [`defaulttreebackend`](@ref) over the plain
`H2Trees.TwoNTree` backend whenever `ParallelKMeans.jl` is loaded.

# Arguments

  - `numberofclusters`: Number of clusters per split (default `2`)
  - `minvalues`: Minimum number of values per leaf before splitting stops, used for
    both `testspace` and `trialspace` (default `200`)
  - `kwargs...`: Forwarded to `H2Trees.KMeansTree`
"""
function AdaptiveCrossApproximation._tree(
    ::AdaptiveCrossApproximation.KMeansTreeBackend,
    testspace::BEAST.Space,
    trialspace::BEAST.Space;
    numberofclusters::Int=2,
    minvalues::Int=200,
    kwargs...,
)
    testtree = H2Trees.KMeansTree(
        BEAST.positions(testspace), numberofclusters; minvalues=minvalues, kwargs...
    )
    trialtree = H2Trees.KMeansTree(
        BEAST.positions(trialspace), numberofclusters; minvalues=minvalues, kwargs...
    )
    return H2Trees.BlockTree(testtree, trialtree)
end

end # module ACAParallelKMeansTrees
