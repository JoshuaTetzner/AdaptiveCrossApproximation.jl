# Forward the common tree interface without renaming backend operations.
AdaptiveCrossApproximation.testtree(tree::H2Trees.BlockTree) = H2Trees.testtree(tree)
AdaptiveCrossApproximation.trialtree(tree::H2Trees.BlockTree) = H2Trees.trialtree(tree)
AdaptiveCrossApproximation.BlockTree(
    testtree::H2Trees.H2ClusterTree, trialtree::H2Trees.H2ClusterTree
) = H2Trees.BlockTree(testtree, trialtree)
AdaptiveCrossApproximation.root(tree::H2Trees.H2ClusterTree) = H2Trees.root(tree)
AdaptiveCrossApproximation.children(tree::H2Trees.H2ClusterTree, node::Int) =
    H2Trees.children(tree, node)
AdaptiveCrossApproximation.parent(tree::H2Trees.H2ClusterTree, node::Int) =
    H2Trees.parent(tree, node)
AdaptiveCrossApproximation.firstchild(tree::H2Trees.H2ClusterTree, node::Int) =
    H2Trees.firstchild(tree, node)
AdaptiveCrossApproximation.isleaf(tree::H2Trees.H2ClusterTree, node::Int) =
    H2Trees.isleaf(tree, node)
AdaptiveCrossApproximation.leaves(tree::H2Trees.H2ClusterTree) = H2Trees.leaves(tree)
AdaptiveCrossApproximation.center(tree::H2Trees.H2ClusterTree, node::Int) =
    H2Trees.center(tree, node)
AdaptiveCrossApproximation.values(
    tree::H2Trees.H2ClusterTree, node::Union{Int,Vector{Int}}
) = H2Trees.values(tree, node)
AdaptiveCrossApproximation.numberofnodes(tree::H2Trees.H2ClusterTree) =
    H2Trees.numberofnodes(tree)
AdaptiveCrossApproximation.numberofvalues(tree::H2Trees.H2ClusterTree, node::Int) =
    H2Trees.numberofvalues(tree, node)
AdaptiveCrossApproximation.levels(tree::H2Trees.H2ClusterTree) = H2Trees.levels(tree)
AdaptiveCrossApproximation.LevelIterator(tree::H2Trees.H2ClusterTree, level::Int) =
    H2Trees.LevelIterator(tree, level)
