function fars!(values, farvalues, diagonalflags, cluster, a::Int, b::Int, isnear, indices)
    if a > b
        return fars!(values, farvalues, diagonalflags, cluster, b, a, isnear, indices)
    end

    if !isnear(cluster, cluster, a, b)
        values[a] = indices(a)
        push!(farvalues[a], indices(b))
        push!(diagonalflags[a], a == b)
    elseif a == b
        children = collect(H2Trees.children(cluster, a))
        for j in eachindex(children), i in 1:j
            fars!(
                values,
                farvalues,
                diagonalflags,
                cluster,
                children[i],
                children[j],
                isnear,
                indices,
            )
        end
    elseif !H2Trees.isleaf(cluster, a) && !H2Trees.isleaf(cluster, b)
        for childa in H2Trees.children(cluster, a), childb in H2Trees.children(cluster, b)
            fars!(
                values, farvalues, diagonalflags, cluster, childa, childb, isnear, indices
            )
        end
    end
    return nothing
end

function AdaptiveCrossApproximation.symmetricfarinteractions(
    tree::H2Trees.BlockTree, isnear::IsNearFunctor, spaceordering::PreserveSpaceOrder
)
    return AdaptiveCrossApproximation.symmetricfarinteractions(
        H2Trees.testtree(tree), isnear, spaceordering
    )
end

function AdaptiveCrossApproximation.symmetricfarinteractions(
    tree::H2Trees.BlockTree, isnear::IsNearFunctor, spaceordering::PermuteSpaceInPlace
)
    return AdaptiveCrossApproximation.symmetricfarinteractions(
        H2Trees.testtree(tree), isnear, spaceordering
    )
end

function AdaptiveCrossApproximation.symmetricfarinteractions(
    cluster::H2Trees.H2ClusterTree, isnear::IsNearFunctor, spaceordering::SpaceOrderingStyle
)
    AdaptiveCrossApproximation._checkcrosslevel(isnear)
    n = length(cluster.nodes)
    values = Vector{Vector{Int}}(undef, n)
    nestedfarvalues = [Vector{Int}[] for _ in 1:n]
    nestedflags = [Bool[] for _ in 1:n]
    indices(node) = _interactionindices(cluster, node, spaceordering)
    fars!(
        values,
        nestedfarvalues,
        nestedflags,
        cluster,
        H2Trees.root(cluster),
        H2Trees.root(cluster),
        isnear,
        indices,
    )
    farptr, farvalues = AdaptiveCrossApproximation.linearizestorage(nestedfarvalues)
    flagptr, flags = AdaptiveCrossApproximation.linearizestorage(nestedflags)
    @assert farptr == flagptr
    return values, farptr, farvalues, flags
end
