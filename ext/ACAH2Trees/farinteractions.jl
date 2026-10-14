import H2Trees: isleaf, testtree, trialtree, root, children, numberofvalues

function fars!(
    treea,
    treeb,
    values::U,
    farvalues::Vector{U},
    tnode::Int,
    snodes::V;
    isnear=AdaptiveCrossApproximation.isnear(1.0),
) where {V<:Vector{Int},U<:Vector{UnitRange{Int}}}
    childnodes = Int[]
    localfarnodes = UnitRange{Int}[]
    for snode in snodes
        if !isnear(treea, treeb, tnode, snode)
            push!(localfarnodes, range(treeb, snode))
        else
            append!(childnodes, collect(children(treeb, snode)))
        end
    end
    farvalues[tnode] = localfarnodes
    !isempty(localfarnodes) && (values[tnode] = range(treea, tnode))
    for child in children(treea, tnode)
        fars!(treea, treeb, values, farvalues, child, childnodes; isnear=isnear)
    end
end

function fars!(
    treea,
    treeb,
    values::VV,
    farvalues::Vector{VV},
    tnode::Int,
    snodes::V;
    isnear=AdaptiveCrossApproximation.isnear(1.0),
) where {V<:Vector{Int},VV<:Vector{V}}
    childnodes = Int[]
    localfarvalues = Vector{Int}[]
    for snode in snodes
        if !isnear(treea, treeb, tnode, snode)
            push!(localfarvalues, H2Trees.values(treeb, snode))
        else
            append!(childnodes, collect(children(treeb, snode)))
        end
    end
    farvalues[tnode] = localfarvalues
    !isempty(localfarvalues) && (values[tnode] = H2Trees.values(treea, tnode))
    for child in children(treea, tnode)
        fars!(treea, treeb, values, farvalues, child, childnodes; isnear=isnear)
    end
end

function AdaptiveCrossApproximation.farinteractions(
    tree::BlockTree, isnear::IsNearFunctor, spaceordering::SpaceOrderingStyle
)
    return AdaptiveCrossApproximation.farinteractions(
        testtree(tree), trialtree(tree), isnear, spaceordering
    )
end

function AdaptiveCrossApproximation.farinteractions(
    treea::H2Trees.H2ClusterTree,
    treeb::H2Trees.H2ClusterTree;
    isnear=AdaptiveCrossApproximation.isnear(),
    spaceordering::SpaceOrderingStyle=PreserveSpaceOrder(),
)
    return AdaptiveCrossApproximation.farinteractions(
        treea, treeb, IsNearFunctor(isnear), spaceordering
    )
end

function AdaptiveCrossApproximation.farinteractions(
    treea::H2Trees.H2ClusterTree,
    treeb::H2Trees.H2ClusterTree,
    isnear::IsNearFunctor,
    ::PreserveSpaceOrder,
)
    AdaptiveCrossApproximation._checkcrosslevel(isnear)
    vals = Vector{Vector{Int}}(undef, length(treea.nodes))
    nestedfarvals = [Vector{Int}[] for _ in eachindex(treea.nodes)]
    if !isnear(treea, treeb, root(treea), root(treeb))
        vals[root(treea)] = H2Trees.values(treea, root(treea))
        nestedfarvals[root(treea)] = [H2Trees.values(treeb, root(treeb))]
    else
        fars!(treea, treeb, vals, nestedfarvals, root(treea), [root(treeb)]; isnear=isnear)
    end
    farptr, farvals = AdaptiveCrossApproximation.linearizestorage(nestedfarvals)
    return vals, farptr, farvals
end

function AdaptiveCrossApproximation.farinteractions(
    treea::H2Trees.H2ClusterTree,
    treeb::H2Trees.H2ClusterTree,
    isnear::IsNearFunctor,
    ::PermuteSpaceInPlace,
)
    AdaptiveCrossApproximation._checkcrosslevel(isnear)
    vals = Vector{UnitRange{Int}}(undef, length(treea.nodes))
    farvals = [UnitRange{Int}[] for _ in eachindex(treea.nodes)]
    if !isnear(treea, treeb, root(treea), root(treeb))
        vals[root(treea)] = range(treea, root(treea))
        farvals[root(treea)] = [range(treeb, root(treeb))]
    else
        fars!(treea, treeb, vals, farvals, root(treea), [root(treeb)]; isnear=isnear)
    end
    farptr, farvals = AdaptiveCrossApproximation.linearizestorage(farvals)
    return vals, farptr, farvals
end
