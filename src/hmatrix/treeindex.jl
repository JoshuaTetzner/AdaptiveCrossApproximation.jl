"""
    IdentityPermutation

Zero-storage marker indicating that an axis is already in tree order.
"""
struct IdentityPermutation end

_permindex(permutation::AbstractVector, i) = @inbounds permutation[i]
_permindex(::IdentityPermutation, i) = i

_clusterranges(tree) = error("Cluster ranges are not implemented for $(typeof(tree))")
_clusterroot(tree) = error("Cluster root is not implemented for $(typeof(tree))")
_clusterchildren(tree, node) =
    error("Cluster children are not implemented for $(typeof(tree))")

_treepermutation(tree, ::PreserveSpaceOrder) = permutation(tree)
_treepermutation(tree, ::PermuteSpaceInPlace) = IdentityPermutation()

# Read-only tree-order analysis of an HMatrixCore. Workspaces share one index:
# `rowperm`/`colperm` map tree positions to external indices, `rowranges` and
# `colranges` map cluster IDs to contiguous tree-order ranges, and `entryrows`
# maps CSR entries to their row clusters.
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

_overlap(a, b) = max(first(a), first(b)):min(last(a), last(b))
_relative(a, origin) = (first(a) - first(origin) + 1):(last(a) - first(origin) + 1)

_rowrange(index, k) = index.rowranges[index.entryrows[k]]
_colrange(index, k) = index.colranges[index.csr.colnodes[k]]

# Workspaces analysed for one core may serve every core with the same tree and
# terminal partition; numerical ranks may differ.
function _checkstructure(index, core::HMatrixCore)
    size(core) == index.dim || throw(DimensionMismatch("matrix dimensions differ"))
    core.tree === index.tree ||
        throw(ArgumentError("matrix uses a different cluster tree than the workspace"))
    a, b = core.csr, index.csr
    a === b ||
        a.rowptr == b.rowptr &&
        a.colnodes == b.colnodes &&
        a.isfar == b.isfar &&
        a.transposed == b.transposed &&
        a.storageidx == b.storageidx ||
        throw(ArgumentError("matrix has a different block partition than the workspace"))
    return nothing
end

# Equal-cost contiguous chunks of `costs`, as pointers into it.
function _costchunks(costs, nchunks)
    total = sum(costs; init=0.0)
    ptr, acc, c = [1], 0.0, 1
    for (i, cost) in enumerate(costs)
        acc += cost
        if c < nchunks && acc >= total * c / nchunks
            push!(ptr, i + 1)
            c += 1
        end
    end
    ptr[end] == length(costs) + 1 || push!(ptr, length(costs) + 1)
    return ptr
end
