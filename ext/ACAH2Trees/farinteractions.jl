import H2Trees: isleaf, testtree, trialtree, root, children, numberofvalues

function fars_consecutive!(
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
    values[tnode] = range(treea, tnode)
    for child in children(treea, tnode)
        fars_consecutive!(treea, treeb, values, farvalues, child, childnodes; isnear=isnear)
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
    !isempty(localfarvalues) && (farvalues[tnode] = localfarvalues)
    !isempty(localfarvalues) && (values[tnode] = H2Trees.values(treea, tnode))
    for child in children(treea, tnode)
        fars!(treea, treeb, values, farvalues, child, childnodes; isnear=isnear)
    end
end

function AdaptiveCrossApproximation.farinteractions(
    tree::BlockTree; isnear=AdaptiveCrossApproximation.isnear(1.0)
)
    return AdaptiveCrossApproximation.farinteractions(
        testtree(tree), trialtree(tree); isnear=isnear
    )
end

function AdaptiveCrossApproximation.farinteractions(
    treea, treeb; isnear=AdaptiveCrossApproximation.isnear(1.0)
)
    vals = Vector{Vector{Int}}(undef, length(treea.nodes))
    nestedfarvals = Vector{Vector{Int}}(undef, length(treea.nodes))
    if !isnear(treea, treeb, root(treea), root(treeb))
        vals[root(treea)] = H2Trees.values(treea, root(treea))
        nestedfarvals[root(treea)] = [H2Trees.values(treeb, root(treeb))]
    else
        fars!(treea, treeb, vals, nestedfarvals, root(treea), [root(treeb)]; isnear=isnear)
    end

    return vals, AdaptiveCrossApproximation.linearizestorage(nestedfarvals)
end

function AdaptiveCrossApproximation.farinteractions_consecutive(
    tree::BlockTree; isnear=AdaptiveCrossApproximation.isnear(1.0)
)
    return AdaptiveCrossApproximation.farinteractions_consecutive(
        testtree(tree), trialtree(tree); isnear=isnear
    )
end

function AdaptiveCrossApproximation.farinteractions_consecutive(
    treea, treeb; isnear=AdaptiveCrossApproximation.isnear(1.0)
)
    vals = Vector{UnitRange{Int}}(undef, length(treea.nodes))
    farvals = Vector{Vector{UnitRange{Int}}}(undef, length(treea.nodes))
    if !isnear(treea, treeb, root(treea), root(treeb))
        farvals[root(treea)] = range(testtree(treea), root(treeb))
        vals[root(treea)] = [range(testtree(treea), root(treea))]
    end
    fars_consecutive!(
        treea, treeb, vals, farvals, root(treea), [root(treeb)]; isnear=isnear
    )
    farptr, farvals = AdaptiveCrossApproximation.linearizestorage(farvals)
    return vals, farptr, farvals
end
