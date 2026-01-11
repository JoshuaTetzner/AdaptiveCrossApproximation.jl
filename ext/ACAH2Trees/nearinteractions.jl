function (isnear::AdaptiveCrossApproximation.H.IsNearFunctor{F})(
    treea::H2Trees.TwoNTree, treeb::H2Trees.TwoNTree, nodea::Int, nodeb::Int
) where {F}
    ths = H2Trees.halfsize(treea, nodea) * sqrt(3)
    shs = H2Trees.halfsize(treeb, nodeb) * sqrt(3)
    dist = norm(H2Trees.center(treea, nodea) - H2Trees.center(treeb, nodeb)) - (ths + shs)

    (2 * max(ths, shs) <= isnear.η * max(dist, 0.0)) ? (return false) : (return true)
end

function (isnear::AdaptiveCrossApproximation.H.IsNearFunctor{F})(
    treea::H2Trees.BoundingBallTree, treeb::H2Trees.BoundingBallTree, nodea::Int, nodeb::Int
) where {F}
    ths = H2Trees.radius(treea, nodea) * sqrt(3)
    shs = H2Trees.radius(treeb, nodeb) * sqrt(3)
    dist = norm(H2Trees.center(treea, nodea) - H2Trees.center(treeb, nodeb)) - (ths + shs)

    (2 * max(ths, shs) <= isnear.η * max(dist, 0.0)) ? (return false) : (return true)
end

function nears!(
    tree,
    values::Vector{V},
    lvalues::Vector{V},
    nearvalues::Vector{V},
    lennearvalues::Vector{V},
    tnode::Int,
    snodes::V;
    isnear=H2Trees.isnear,
) where {V<:Vector{Int}}
    tnodes = Int[]
    nearnodes = Int[]
    childnearnodes = Int[]
    for snode in snodes
        if isnear(testtree(tree), trialtree(tree), tnode, snode)
            if isleaf(testtree(tree), tnode) || isleaf(trialtree(tree), snode)
                push!(nearnodes, snode)
                push!(tnodes, tnode)
            else
                append!(childnearnodes, collect(children(trialtree(tree), snode)))
            end
        end
    end
    if nearnodes != []
        push!(nearvalues, H2Trees.values(trialtree(tree), nearnodes)[1])
        push!(nearvalues, length(H2Trees.values(trialtree(tree), nearnodes)))
        push!(values, H2Trees.values(testtree(tree), tnodes)[1])
        push!(lvalues, length(H2Trees.values(testtree(tree), tnodes)))
    end
    if childnearnodes != []
        for child in children(testtree(tree), tnode)
            nears!(
                tree,
                values,
                nearvalues,
                lvalues,
                lennearvalues,
                child,
                childnearnodes;
                isnear=isnear,
            )
        end
    end
end

function nearinteractions(tree::H2Trees.BlockTree; isnear=H2Trees.isnear)
    !isnear(testtree(tree), trialtree(tree), root(testtree(tree)), root(trialtree(tree))) &&
        return Vector{Int}(), Vector{Int}[]
    values = Vector{Int}[]
    lvalues = Vector{Int}[]
    nearvalues = Vector{Int}[]
    lnearvalues = Vector{Int}[]
    nears!(
        tree,
        values,
        lvalues,
        nearvalues,
        lnearvalues,
        root(testtree(tree)),
        [root(trialtree(tree))];
        isnear=isnear,
    )
    return values, nearvalues
end
