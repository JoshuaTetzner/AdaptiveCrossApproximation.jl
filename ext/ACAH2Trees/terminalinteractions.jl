function _terminalinteractions!(
    rownodes, colnodes, farflags, rowtree, coltree, rownode::Int, colnode::Int, predicate
)
    if !predicate(rowtree, coltree, rownode, colnode)
        push!(rownodes, rownode)
        push!(colnodes, colnode)
        push!(farflags, true)
    elseif H2Trees.isleaf(rowtree, rownode) || H2Trees.isleaf(coltree, colnode)
        push!(rownodes, rownode)
        push!(colnodes, colnode)
        push!(farflags, false)
    else
        for rowchild in H2Trees.children(rowtree, rownode)
            for colchild in H2Trees.children(coltree, colnode)
                _terminalinteractions!(
                    rownodes,
                    colnodes,
                    farflags,
                    rowtree,
                    coltree,
                    rowchild,
                    colchild,
                    predicate,
                )
            end
        end
    end
    return nothing
end

function AdaptiveCrossApproximation.terminalinteractions(
    tree::H2Trees.BlockTree, predicate::IsNearFunctor, ::SpaceOrderingStyle, ::Val{false}
)
    AdaptiveCrossApproximation._checkcrosslevel(predicate)
    rownodes = Int[]
    colnodes = Int[]
    farflags = BitVector()
    rowtree = H2Trees.testtree(tree)
    coltree = H2Trees.trialtree(tree)
    _terminalinteractions!(
        rownodes,
        colnodes,
        farflags,
        rowtree,
        coltree,
        H2Trees.root(rowtree),
        H2Trees.root(coltree),
        predicate,
    )
    nblocks = length(rownodes)
    return rownodes, colnodes, farflags, falses(nblocks), collect(1:nblocks)
end

function _symmetricterminalinteractions!(
    rownodes, colnodes, farflags, tree, rownode::Int, colnode::Int, predicate
)
    rownode > colnode && return _symmetricterminalinteractions!(
        rownodes, colnodes, farflags, tree, colnode, rownode, predicate
    )

    if !predicate(tree, tree, rownode, colnode)
        push!(rownodes, rownode)
        push!(colnodes, colnode)
        push!(farflags, true)
    elseif H2Trees.isleaf(tree, rownode) || H2Trees.isleaf(tree, colnode)
        push!(rownodes, rownode)
        push!(colnodes, colnode)
        push!(farflags, false)
    elseif rownode == colnode
        children = collect(H2Trees.children(tree, rownode))
        for j in eachindex(children), i in 1:j
            _symmetricterminalinteractions!(
                rownodes, colnodes, farflags, tree, children[i], children[j], predicate
            )
        end
    else
        for rowchild in H2Trees.children(tree, rownode)
            for colchild in H2Trees.children(tree, colnode)
                _symmetricterminalinteractions!(
                    rownodes, colnodes, farflags, tree, rowchild, colchild, predicate
                )
            end
        end
    end
    return nothing
end

function AdaptiveCrossApproximation.terminalinteractions(
    tree::H2Trees.BlockTree, predicate::IsNearFunctor, ::SpaceOrderingStyle, ::Val{true}
)
    AdaptiveCrossApproximation._checkcrosslevel(predicate)
    cluster = H2Trees.testtree(tree)
    storedrows = Int[]
    storedcols = Int[]
    storedfar = BitVector()
    _symmetricterminalinteractions!(
        storedrows,
        storedcols,
        storedfar,
        cluster,
        H2Trees.root(cluster),
        H2Trees.root(cluster),
        predicate,
    )

    rownodes = Int[]
    colnodes = Int[]
    farflags = BitVector()
    transposed = BitVector()
    storagekeys = Int[]
    for key in eachindex(storedrows)
        rownode = storedrows[key]
        colnode = storedcols[key]
        push!(rownodes, rownode)
        push!(colnodes, colnode)
        push!(farflags, storedfar[key])
        push!(transposed, false)
        push!(storagekeys, key)
        if rownode != colnode
            push!(rownodes, colnode)
            push!(colnodes, rownode)
            push!(farflags, storedfar[key])
            push!(transposed, true)
            push!(storagekeys, key)
        end
    end
    return rownodes, colnodes, farflags, transposed, storagekeys
end

AdaptiveCrossApproximation.clusterindices(
    tree::H2Trees.H2ClusterTree, node::Int, ::PreserveSpaceOrder
) = H2Trees.values(tree, node)

AdaptiveCrossApproximation.clusterindices(
    tree::H2Trees.H2ClusterTree, node::Int, ::PermuteSpaceInPlace
) = collect(range(tree, node))

AdaptiveCrossApproximation.numberofnodes(tree::H2Trees.H2ClusterTree) =
    H2Trees.numberofnodes(tree)

# Postorder construction counts leaf values once. numberofvalues on every
# internal node would repeatedly traverse its subtree. Original indices are
# deliberately not interpreted as tree-order offsets.
function _clusterranges!(ranges, tree, node, start)
    stop = start
    if H2Trees.isleaf(tree, node)
        stop += H2Trees.numberofvalues(tree, node)
    else
        for child in H2Trees.children(tree, node)
            stop = _clusterranges!(ranges, tree, child, stop)
        end
    end
    ranges[node] = start:(stop - 1)
    return stop
end

function AdaptiveCrossApproximation._clusterranges(tree::H2Trees.H2ClusterTree)
    ranges = fill(1:0, H2Trees.numberofnodes(tree))
    _clusterranges!(ranges, tree, H2Trees.root(tree), 1)
    return ranges
end

AdaptiveCrossApproximation._clusterroot(tree::H2Trees.H2ClusterTree) = H2Trees.root(tree)
AdaptiveCrossApproximation._clusterchildren(tree::H2Trees.H2ClusterTree, node) =
    H2Trees.children(tree, node)

function AdaptiveCrossApproximation._symmetrictree(tree::H2Trees.BlockTree)
    cluster = H2Trees.testtree(tree)
    return H2Trees.BlockTree(cluster, cluster)
end
