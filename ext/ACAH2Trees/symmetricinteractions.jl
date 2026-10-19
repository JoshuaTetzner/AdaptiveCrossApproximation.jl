# Unordered terminal pairs `a <= b` of one cluster tree: an admissible pair is a
# far block, a near pair is a near block once one side is a leaf. Otherwise a self
# pair descends into all pairs of its children and a pair of distinct clusters
# into the products of their children.
function symmetricinteractions!(nearnodes, farnodes, cluster, a::Int, b::Int, isnear)
    a > b && return symmetricinteractions!(nearnodes, farnodes, cluster, b, a, isnear)
    if !isnear(cluster, cluster, a, b)
        push!(farnodes[a], b)
    elseif isleaf(cluster, a) || isleaf(cluster, b)
        push!(nearnodes[a], b)
    elseif a == b
        childnodes = collect(children(cluster, a))
        for j in eachindex(childnodes), i in 1:j
            symmetricinteractions!(
                nearnodes, farnodes, cluster, childnodes[i], childnodes[j], isnear
            )
        end
    else
        for childa in children(cluster, a), childb in children(cluster, b)
            symmetricinteractions!(nearnodes, farnodes, cluster, childa, childb, isnear)
        end
    end
    return nothing
end

function AdaptiveCrossApproximation.symmetricinteractions(tree::BlockTree, isnear)
    AdaptiveCrossApproximation._checkcrosslevel(isnear)
    cluster = testtree(tree)
    nnodes = H2Trees.numberofnodes(cluster)
    nearnodes = [Int[] for _ in 1:nnodes]
    farnodes = [Int[] for _ in 1:nnodes]
    symmetricinteractions!(
        nearnodes, farnodes, cluster, root(cluster), root(cluster), isnear
    )
    return AdaptiveCrossApproximation._interactions(nearnodes, farnodes)
end
