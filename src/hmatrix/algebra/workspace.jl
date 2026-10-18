# Tree-order arithmetic view of an HMatrixCore. The block vectors hold
# references to the bound core's numerical blocks: arithmetic through the view
# overwrites that core. Every plan built on a view addresses blocks only through
# these vectors, so `_bind!` moves the view, and all its plans, to another core
# with the same tree and partition. Arithmetic requires full (unshared,
# untransposed) block storage.
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

function _TreeView(index::_TreeIndex, core::HMatrixCore{T}) where {T}
    _checkstructure(index, core)
    _checkfullstorage(core)
    return _TreeView(
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

function _bind!(W::_TreeView, core::HMatrixCore)
    _checkstructure(W, core)
    _checkfullstorage(core)
    copyto!(W.near, core.near)
    copyto!(W.far, core.far)
    return W
end

Base.eltype(::Type{<:_TreeView{T}}) where {T} = T
Base.eltype(::_TreeView{T}) where {T} = T
Base.size(W::_TreeView) = W.dim
Base.size(W::_TreeView, dim::Integer) = W.dim[dim]
