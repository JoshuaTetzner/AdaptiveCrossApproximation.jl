function fars!(
    treea, treeb, farnodes::Vector{V}, tnode::Int, snodes::V; isnear=H2Trees.isnear
) where {V<:Vector{Int}}
    childnodes = Int[]
    localfarnodes = Int[]
    for snode in snodes
        if !isnear(treea, treeb, tnode, snode)
            push!(localfarnodes, snode)
        else
            append!(childnodes, collect(children(treeb, snode)))
        end
    end
    farnodes[tnode] = localfarnodes
    if childnodes != []
        for child in children(treea, tnode)
            fars!(treea, treeb, farnodes, child, childnodes; isnear=isnear)
        end
    end
end

function farinteractions(tree; isnear=H2Trees.isnear)
    if !isnear(testtree(tree), trialtree(tree), root(testtree(tree)), root(trialtree(tree)))
        values = H2Trees.values(testtree(tree), root(testtree(tree)))
        lvalues = length(H2Trees.values(testtree(tree), root(testtree(tree))))
        farvalues = H2Trees.values(trialtree(tree), root(trialtree(tree)))
        lfarvalues = length(H2Trees.values(trialtree(tree), root(trialtree(tree))))
        return [values], [lvalues], [farvalues], [lfarvalues]
    end
    farnodes = Vector{Vector{Int}}(undef, length(d.nodes))
    fars!(
        testtree(tree),
        trialtree(tree),
        farnodes,
        root(testtree(tree)),
        [root(trialtree(tree))];
        isnear=isnear,
    )

    vals = Vector{Vector{Int}}[]
    farvals = Vector{Vector{Int}}[]
    lvals = Vector{Vector{Int}}[]
    lfarvals = Vector{Vector{Int}}[]
    for level in level(testtree(tree))
        levvals = Vector{Int}[]
        levfarvals = Vector{Int}[]
        levlvals = Vector{Int}[]
        levlfarvals = Vector{Int}[]
        for node in LevelIterator(testtree(tree), level)
            if farnodes[node] != []
                push!(
                    levvals,
                    [H2Trees.values(testtree(tree), node)[1] for _ in farnodes[node]],
                )
                push!(
                    levlvals,
                    [length(H2Trees.values(testtree(tree), node)) for _ in farnodes[node]],
                )
                push!(
                    levfarvals,
                    [
                        H2Trees.values(trialtree(tree), farnode)[1] for
                        farnode in farnodes[node]
                    ],
                )
                push!(
                    levlfarvals,
                    [
                        length(H2Trees.values(trialtree(tree), farnode)) for
                        farnode in farnodes[node]
                    ],
                )
            end
        end
        push!(vals, levvals)
        push!(farvals, levfarvals)
        push!(lvals, levlvals)
        push!(lfarvals, levlfarvals)
    end
    return vals, lvals, farvals, lfarvals
end
