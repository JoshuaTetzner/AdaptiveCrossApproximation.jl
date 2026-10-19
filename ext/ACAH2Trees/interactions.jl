function (isnear::AdaptiveCrossApproximation.IsNearFunctor{F})(
    treea::H2Trees.TwoNTree, treeb::H2Trees.TwoNTree, nodea::Int, nodeb::Int
) where {F<:Real}
    ths = H2Trees.halfsize(treea, nodea) * sqrt(3)
    shs = H2Trees.halfsize(treeb, nodeb) * sqrt(3)
    dist = norm(H2Trees.center(treea, nodea) - H2Trees.center(treeb, nodeb)) - (ths + shs)

    (2 * max(ths, shs) <= isnear.η * max(dist, 0.0)) ? (return false) : (return true)
end

function (isnear::AdaptiveCrossApproximation.IsNearFunctor{F})(
    treea::H2Trees.BoundingBallTree, treeb::H2Trees.BoundingBallTree, nodea::Int, nodeb::Int
) where {F<:Real}
    ths = H2Trees.radius(treea, nodea)
    shs = H2Trees.radius(treeb, nodeb)
    dist = norm(H2Trees.center(treea, nodea) - H2Trees.center(treeb, nodeb)) - (ths + shs)
    maxhs = max(ths, shs)
    sep = max(dist, 0.0)

    # A leaf containing a single point (or coincident points) has radius 0. Two
    # such leaves at zero separation (e.g. a node against itself) then satisfy
    # 2*maxhs <= η*sep as 0 <= 0, so they get classified as admissible (far)
    # rather than near.
    if iszero(maxhs) && iszero(sep)
        @warn("Zero-radius node pair at zero separation classified as far.", maxlog = 1)
    end

    (2 * maxhs <= isnear.η * sep) ? (return false) : (return true)
end

# Terminal blocks below test node `tnode`: `snodes` are the trial nodes still to
# be classified against it. An admissible pair is a far block, a near pair is a
# near block once one side is a leaf; otherwise the trial children are classified
# against the test children.
function interactions!(nearnodes, farnodes, tree, tnode::Int, snodes::Vector{Int}, isnear)
    childnodes = Int[]
    for snode in snodes
        if !isnear(testtree(tree), trialtree(tree), tnode, snode)
            push!(farnodes[tnode], snode)
        elseif isleaf(testtree(tree), tnode) || isleaf(trialtree(tree), snode)
            push!(nearnodes[tnode], snode)
        else
            append!(childnodes, children(trialtree(tree), snode))
        end
    end
    isempty(childnodes) && return nothing
    for child in children(testtree(tree), tnode)
        interactions!(nearnodes, farnodes, tree, child, childnodes, isnear)
    end
    return nothing
end

function AdaptiveCrossApproximation.interactions(tree::BlockTree, isnear)
    AdaptiveCrossApproximation._checkcrosslevel(isnear)
    nnodes = H2Trees.numberofnodes(testtree(tree))
    nearnodes = [Int[] for _ in 1:nnodes]
    farnodes = [Int[] for _ in 1:nnodes]
    interactions!(
        nearnodes, farnodes, tree, root(testtree(tree)), [root(trialtree(tree))], isnear
    )
    return AdaptiveCrossApproximation._interactions(nearnodes, farnodes)
end
