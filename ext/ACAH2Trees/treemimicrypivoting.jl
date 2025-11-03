function AdaptiveCrossApproximation.center(tree::H2Trees.H2ClusterTree, node::Int)
    return H2Trees.center(tree, node)
end

function AdaptiveCrossApproximation.values(tree::H2Trees.H2ClusterTree, node::Int)
    return H2Trees.values(tree, node)
end

function AdaptiveCrossApproximation.children(tree::H2Trees.H2ClusterTree, node::Int)
    return H2Trees.children(tree, node)
end

function AdaptiveCrossApproximation.firstchild(tree::H2Trees.H2ClusterTree, node::Int)
    return H2Trees.firstchild(tree, node)
end
