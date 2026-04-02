function (isnear::AdaptiveCrossApproximation.IsNearFunctor{F})(
    treea::H2Trees.TwoNTree, treeb::H2Trees.TwoNTree, nodea::Int, nodeb::Int
) where {F}
    ths = H2Trees.halfsize(treea, nodea) * sqrt(3)
    shs = H2Trees.halfsize(treeb, nodeb) * sqrt(3)
    dist = norm(H2Trees.center(treea, nodea) - H2Trees.center(treeb, nodeb)) - (ths + shs)

    (2 * max(ths, shs) <= isnear.η * max(dist, 0.0)) ? (return false) : (return true)
end

function (isnear::AdaptiveCrossApproximation.IsNearFunctor{F})(
    treea::H2Trees.BoundingBallTree, treeb::H2Trees.BoundingBallTree, nodea::Int, nodeb::Int
) where {F}
    ths = H2Trees.radius(treea, nodea)
    shs = H2Trees.radius(treeb, nodeb)
    dist = norm(H2Trees.center(treea, nodea) - H2Trees.center(treeb, nodeb)) - (ths + shs)

    (2 * max(ths, shs) <= isnear.η * max(dist, 0.0)) ? (return false) : (return true)
end

function nears_consecutive!(
    tree,
    values::Vector{R},
    nearvalues::Vector{V},
    hasnears::Vector{Bool},
    tnode::Int,
    snodes::Vector{Int};
    isnear=AdaptiveCrossApproximation.isnear(1.0),
) where {R<:UnitRange{Int},V<:Vector{R}}
    localnearrange = UnitRange{Int}[]
    childnearnodes = Int[]

    for snode in snodes
        if isnear(testtree(tree), trialtree(tree), tnode, snode)
            if isleaf(testtree(tree), tnode) || isleaf(trialtree(tree), snode)
                push!(localnearrange, range(trialtree(tree), snode))
            else
                append!(childnearnodes, collect(children(trialtree(tree), snode)))
            end
        end
    end

    if !isempty(localnearrange)
        values[tnode] = range(testtree(tree), tnode)
        nearvalues[tnode] = localnearrange
        hasnears[tnode] = true
    end

    if !isempty(childnearnodes)
        for child in children(testtree(tree), tnode)
            nears_consecutive!(
                tree, values, nearvalues, hasnears, child, childnearnodes; isnear=isnear
            )
        end
    end
end

function nears!(
    tree,
    values::Vector{V},
    nearvalues::Vector{V},
    tnode::Int,
    snodes::Vector{Int};
    isnear=AdaptiveCrossApproximation.isnear(1.0),
) where {V<:Vector{Int}}
    localnearvalues = Int[]
    childnearnodes = Int[]

    for snode in snodes
        if isnear(testtree(tree), trialtree(tree), tnode, snode)
            if isleaf(testtree(tree), tnode) || isleaf(trialtree(tree), snode)
                append!(localnearvalues, H2Trees.values(trialtree(tree), snode))
            else
                append!(childnearnodes, collect(children(trialtree(tree), snode)))
            end
        end
    end

    if !isempty(localnearvalues)
        values[tnode] = H2Trees.values(testtree(tree), tnode)
        nearvalues[tnode] = localnearvalues
    end

    if !isempty(childnearnodes)
        for child in children(testtree(tree), tnode)
            nears!(tree, values, nearvalues, child, childnearnodes; isnear=isnear)
        end
    end
end

function AdaptiveCrossApproximation.nearinteractions(
    tree::H2Trees.BlockTree; isnear=AdaptiveCrossApproximation.isnear(1.0)
)
    !isnear(testtree(tree), trialtree(tree), root(testtree(tree)), root(trialtree(tree))) &&
        return Vector{Int}[], Vector{Int}[]
    values = Vector{Vector{Int}}(undef, length(testtree(tree).nodes))
    nearvalues = Vector{Vector{Int}}(undef, length(testtree(tree).nodes))
    nears!(
        tree,
        values,
        nearvalues,
        root(testtree(tree)),
        [root(trialtree(tree))];
        isnear=isnear,
    )
    return AdaptiveCrossApproximation.collectassigned(values),
    AdaptiveCrossApproximation.collectassigned(nearvalues)
end

function AdaptiveCrossApproximation.nearinteractions_consecutive(
    tree::H2Trees.BlockTree; isnear=AdaptiveCrossApproximation.isnear(1.0)
)
    !isnear(testtree(tree), trialtree(tree), root(testtree(tree)), root(trialtree(tree))) &&
        return UnitRange{Int}[], Vector{UnitRange{Int}}[]
    values = Vector{UnitRange{Int}}(undef, length(testtree(tree).nodes))
    nearvalues = Vector{Vector{UnitRange{Int}}}(undef, length(testtree(tree).nodes))
    hasnear = zeros(Bool, length(testtree(tree).nodes))
    nears_consecutive!(
        tree,
        values,
        nearvalues,
        hasnear,
        root(testtree(tree)),
        [root(trialtree(tree))];
        isnear=isnear,
    )
    values, nearvalues = AdaptiveCrossApproximation.collectnears(
        values, nearvalues, hasnear
    )
    return values, nearvalues
end
