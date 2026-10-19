"""
    _TreeView(H::HMatrixCore)

Tree-order view of `H` on which all hierarchical arithmetic operates. Rows and
columns are numbered in cluster-tree order, so every cluster is a contiguous
range (`rowranges`, `colranges`), and `entryrows` maps each CSR entry to its row
cluster. `rowperm` and `colperm` map tree positions to the external indices.

The vectors `near` and `far` reference the blocks of the bound matrix, so
arithmetic through the view overwrites that matrix. Plans address blocks only
through these vectors and the CSR storage indices: [`_bind!`](@ref) moves a
view, and with it every plan built on it, to another matrix with the same trees
and block partition. Requires full (not symmetric half-) storage.
"""
struct _TreeView{T,Tree,RowPermutation,ColPermutation}
    tree::Tree
    rowperm::RowPermutation
    colperm::ColPermutation
    near::Vector{Matrix{T}}
    far::Vector{LowRankMatrix{T}}
    csr::BlockCSR{Int}
    csc::BlockCSC{Int}
    rowranges::Vector{UnitRange{Int}}
    colranges::Vector{UnitRange{Int}}
    entryrows::Vector{Int}
    dim::Tuple{Int,Int}
end

function _checkfullstorage(core::HMatrixCore)
    csr = core.csr
    length(csr) == length(core.near) + length(core.far) && !any(csr.transposed) || throw(
        ArgumentError(
            "H-matrix arithmetic requires full block storage; symmetric " *
            "(half-stored) matrices are not supported",
        ),
    )
    return nothing
end

# Stored blocks must match their clusters.
function _checkblocks(W)
    csr = W.csr
    for k in eachindex(csr.storageidx)
        block = csr.isfar[k] ? W.far[csr.storageidx[k]] : W.near[csr.storageidx[k]]
        size(block) == (length(_rowrange(W, k)), length(_colrange(W, k))) ||
            throw(DimensionMismatch("stored block dimensions differ from clusters"))
        block isa LowRankMatrix &&
            size(block.U, 2) != size(block.V, 1) &&
            throw(DimensionMismatch("incompatible low-rank factors"))
    end
    return nothing
end

function _TreeView(index::_TreeIndex, core::HMatrixCore{T}) where {T}
    _checkstructure(index, core)
    _checkfullstorage(core)
    W = _TreeView(
        index.tree,
        index.rowperm,
        index.colperm,
        copy(core.near),
        copy(core.far),
        index.csr,
        index.csc,
        index.rowranges,
        index.colranges,
        index.entryrows,
        index.dim,
    )
    _checkblocks(W)
    return W
end
_TreeView(core::HMatrixCore) = _TreeView(_TreeIndex(core), core)

# A second view on the same index and blocks, for plans that must not move with
# the original view.
_copyview(W::_TreeView) = _TreeView(
    W.tree,
    W.rowperm,
    W.colperm,
    copy(W.near),
    copy(W.far),
    W.csr,
    W.csc,
    W.rowranges,
    W.colranges,
    W.entryrows,
    W.dim,
)

"""
    _bind!(W::_TreeView, H::HMatrixCore)

Point the block vectors of `W` to the blocks of `H`, which must have the same
trees and block partition as the matrix `W` was built for; ranks may differ.
Plans built on `W` then operate on `H` without being rebuilt.
"""
function _bind!(W::_TreeView, core::HMatrixCore)
    _checkstructure(W, core)
    _checkfullstorage(core)
    copyto!(W.near, core.near)
    copyto!(W.far, core.far)
    _checkblocks(W)
    return W
end

Base.eltype(::Type{<:_TreeView{T}}) where {T} = T
Base.eltype(::_TreeView{T}) where {T} = T
Base.size(W::_TreeView) = W.dim
Base.size(W::_TreeView, dim::Integer) = W.dim[dim]

# Largest rank of the far blocks among `entries`.
function _maxrank(W, entries)
    r = 0
    for k in entries
        W.csr.isfar[k] && (r = max(r, rank(W.far[W.csr.storageidx[k]])))
    end
    return r
end
