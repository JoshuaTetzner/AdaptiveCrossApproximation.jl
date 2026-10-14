function nears!(
    diagonals::Vector{Vector{Int}},
    offrows::Vector{Vector{Int}},
    offcols::Vector{Vector{Int}},
    cluster,
    a::Int,
    b::Int,
    isnear,
    indices,
)
    if a > b
        return nears!(diagonals, offrows, offcols, cluster, b, a, isnear, indices)
    end
    isnear(cluster, cluster, a, b) || return nothing

    if a == b
        if H2Trees.isleaf(cluster, a)
            diagonals[a] = indices(a)
        else
            children = collect(H2Trees.children(cluster, a))
            for j in eachindex(children), i in 1:j
                nears!(
                    diagonals,
                    offrows,
                    offcols,
                    cluster,
                    children[i],
                    children[j],
                    isnear,
                    indices,
                )
            end
        end
    elseif H2Trees.isleaf(cluster, a) || H2Trees.isleaf(cluster, b)
        if !isassigned(offrows, a)
            offrows[a] = indices(a)
            offcols[a] = Int[]
        end
        append!(offcols[a], indices(b))
    else
        for childa in H2Trees.children(cluster, a), childb in H2Trees.children(cluster, b)
            nears!(diagonals, offrows, offcols, cluster, childa, childb, isnear, indices)
        end
    end
    return nothing
end

function AdaptiveCrossApproximation.symmetricnearinteractions(
    tree::H2Trees.BlockTree, isnear::IsNearFunctor, spaceordering::PreserveSpaceOrder
)
    return AdaptiveCrossApproximation.symmetricnearinteractions(
        H2Trees.testtree(tree), isnear, spaceordering
    )
end

function AdaptiveCrossApproximation.symmetricnearinteractions(
    tree::H2Trees.BlockTree, isnear::IsNearFunctor, spaceordering::PermuteSpaceInPlace
)
    return AdaptiveCrossApproximation.symmetricnearinteractions(
        H2Trees.testtree(tree), isnear, spaceordering
    )
end

function AdaptiveCrossApproximation.symmetricnearinteractions(
    cluster::H2Trees.H2ClusterTree, isnear::IsNearFunctor, spaceordering::SpaceOrderingStyle
)
    AdaptiveCrossApproximation._checkcrosslevel(isnear)
    n = length(cluster.nodes)
    diagonals = Vector{Vector{Int}}(undef, n)
    offrows = Vector{Vector{Int}}(undef, n)
    offcols = Vector{Vector{Int}}(undef, n)
    indices(node) = _interactionindices(cluster, node, spaceordering)
    nears!(
        diagonals,
        offrows,
        offcols,
        cluster,
        H2Trees.root(cluster),
        H2Trees.root(cluster),
        isnear,
        indices,
    )
    return AdaptiveCrossApproximation.collectassigned(diagonals),
    AdaptiveCrossApproximation.collectassigned(offrows),
    AdaptiveCrossApproximation.collectassigned(offcols)
end
