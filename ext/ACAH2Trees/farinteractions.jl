import H2Trees: isleaf, testtree, trialtree, root, children

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
) where {V<:Vector{Int},VV<:Vector{Vector{Int}}}
    childnodes = Int[]
    localfarnodes = Vector{Int}[]
    for snode in snodes
        if !isnear(treea, treeb, tnode, snode)
            push!(localfarnodes, H2Trees.values(treeb, snode))
        else
            append!(childnodes, collect(children(treeb, snode)))
        end
    end
    farvalues[tnode] = localfarnodes
    values[tnode] = H2Trees.values(treea, tnode)
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
    values = Vector{Vector{Int}}(undef, length(treea.nodes))
    farvalues = Vector{Vector{Vector{Int}}}(undef, length(treea.nodes))
    if !isnear(treea, treeb, root(treea), root(treeb))
        farvalues[root(treea)] = H2Trees.values(testtree(treea), root(treeb))
        values[root(treea)] = H2Trees.values(testtree(treea), root(treea))
        return [1; cumsum(length.(farvalues)) .+ 1], values, reduce(vcat, farvalues)
    end
    fars!(treea, treeb, values, farvalues, root(treea), [root(treeb)]; isnear=isnear)
    return [1; cumsum(length.(farvalues)) .+ 1], values, reduce(vcat, farvalues)
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
    values = Vector{UnitRange{Int}}(undef, length(treea.nodes))
    farvalues = Vector{Vector{UnitRange{Int}}}(undef, length(treea.nodes))
    if !isnear(treea, treeb, root(treea), root(treeb))
        farvalues[root(treea)] = range(testtree(treea), root(treeb))
        values[root(treea)] = range(testtree(treea), root(treea))
        return [1; cumsum(length.(farvalues)) .+ 1], values, reduce(vcat, farvalues)
    end
    fars_consecutive!(treea, treeb, values, farvalues, root(treea), [root(treeb)]; isnear=isnear)
    return [1; cumsum(length.(farvalues)) .+ 1], values, reduce(vcat, farvalues)
end

