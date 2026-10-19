"""
    IdentityPermutation

Zero-storage marker indicating that an axis is already in tree order.
"""
struct IdentityPermutation end

_permindex(permutation::AbstractVector, i) = @inbounds permutation[i]
_permindex(::IdentityPermutation, i) = i

# ACA's contiguous tree-order ranges are not stored by H2Trees. Compute them
# once from the common topology interface, without materializing internal values.
function _clusterranges!(ranges, tree, node, start)
    stop = start
    if isleaf(tree, node)
        stop += numberofvalues(tree, node)
    else
        for child in children(tree, node)
            stop = _clusterranges!(ranges, tree, child, stop)
        end
    end
    ranges[node] = start:(stop - 1)
    return stop
end

function _clusterranges(tree)
    ranges = fill(1:0, numberofnodes(tree))
    _clusterranges!(ranges, tree, root(tree), 1)
    return ranges
end

_treepermutation(tree, ::PreserveSpaceOrder) = collect(values(tree, root(tree)))
_treepermutation(tree, ::PermuteSpaceInPlace) = IdentityPermutation()

"""
    _TreeIndex(H::HMatrixCore)

Read-only tree-order analysis of `H`, shared by the workspaces built from it.
`rowperm` and `colperm` map tree positions to external indices
([`IdentityPermutation`](@ref) for spaces already in tree order), `rowranges`
and `colranges` map cluster IDs to contiguous tree-order ranges, and `entryrows`
maps CSR entries to their row clusters.
"""
struct _TreeIndex{Tree,RowPermutation,ColPermutation}
    tree::Tree
    csr::BlockCSR{Int}
    csc::BlockCSC{Int}
    rowperm::RowPermutation
    colperm::ColPermutation
    rowranges::Vector{UnitRange{Int}}
    colranges::Vector{UnitRange{Int}}
    entryrows::Vector{Int}
    dim::Tuple{Int,Int}
end

function _TreeIndex(core::HMatrixCore)
    rowtree, coltree = testtree(core.tree), trialtree(core.tree)
    rowperm = _treepermutation(rowtree, core.ordering)
    colperm = rowtree === coltree ? rowperm : _treepermutation(coltree, core.ordering)
    rowranges = _clusterranges(rowtree)
    colranges = rowtree === coltree ? rowranges : _clusterranges(coltree)
    csr = core.csr
    entryrows = Vector{Int}(undef, length(csr))
    for row in 1:(length(csr.rowptr) - 1)
        for k in csr.rowptr[row]:(csr.rowptr[row + 1] - 1)
            entryrows[k] = row
        end
    end
    return _TreeIndex(
        core.tree,
        csr,
        core.csc,
        rowperm,
        colperm,
        rowranges,
        colranges,
        entryrows,
        size(core),
    )
end

_rowrange(index, k) = index.rowranges[index.entryrows[k]]
_colrange(index, k) = index.colranges[index.csr.colnodes[k]]
