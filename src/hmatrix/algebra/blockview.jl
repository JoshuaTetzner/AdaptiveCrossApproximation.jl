# Non-owning tree-order block of a `_TreeView`. Integer arguments of `blockview`
# select cluster nodes; ranges select tree-order positions. Construction caches
# the contributing CSR entries, found through the cluster trees and CSR/CSC
# indices; ancestor blocks are restricted, not materialized. Numerical factors
# and ranks may change between uses, the partition may not. Full views need no
# entry vector.
struct HMatrixBlock{T,W<:_TreeView{T},I<:AbstractVector{Int}}
    workspace::W
    rows::UnitRange{Int}
    cols::UnitRange{Int}
    entries::I
end

Base.size(H::HMatrixBlock) = (length(H.rows), length(H.cols))
Base.size(H::HMatrixBlock, dim::Integer) = size(H)[dim]
Base.eltype(::HMatrixBlock{T}) where {T} = T

function _planrowentries!(entries, W, node, rows, cols)
    isempty(_overlap(W.rowranges[node], rows)) && return entries
    csr = W.csr
    for k in csr.rowptr[node]:(csr.rowptr[node + 1] - 1)
        isempty(_overlap(W.colranges[csr.colnodes[k]], cols)) || push!(entries, k)
    end
    for child in _clusterchildren(testtree(W.tree), node)
        _planrowentries!(entries, W, child, rows, cols)
    end
    return entries
end

function _plancolentries!(entries, W, node, rows, cols)
    isempty(_overlap(W.colranges[node], cols)) && return entries
    csc = W.csc
    for j in csc.colptr[node]:(csc.colptr[node + 1] - 1)
        isempty(_overlap(W.rowranges[csc.rownodes[j]], rows)) ||
            push!(entries, csc.csrmap[j])
    end
    for child in _clusterchildren(trialtree(W.tree), node)
        _plancolentries!(entries, W, child, rows, cols)
    end
    return entries
end

function blockview(W::_TreeView{T}) where {T}
    entries = Base.OneTo(length(W.csr))
    return HMatrixBlock{T,typeof(W),typeof(entries)}(W, 1:size(W, 1), 1:size(W, 2), entries)
end

function blockview(W::_TreeView{T}, rows::UnitRange{Int}, cols::UnitRange{Int}) where {T}
    1 <= first(rows) <= size(W, 1) + 1 &&
    0 <= last(rows) <= size(W, 1) &&
    1 <= first(cols) <= size(W, 2) + 1 &&
    0 <= last(cols) <= size(W, 2) || throw(BoundsError(W, (rows, cols)))
    rows == (1:size(W, 1)) && cols == (1:size(W, 2)) && return blockview(W)
    entries = Int[]
    if !isempty(rows) && !isempty(cols)
        if Float64(length(rows)) * size(W, 2) <= Float64(length(cols)) * size(W, 1)
            _planrowentries!(entries, W, _clusterroot(testtree(W.tree)), rows, cols)
        else
            _plancolentries!(entries, W, _clusterroot(trialtree(W.tree)), rows, cols)
        end
        sort!(entries)
    end
    return HMatrixBlock{T,typeof(W),typeof(entries)}(W, rows, cols, entries)
end

blockview(W::_TreeView, rownode::Int, colnode::Int) =
    blockview(W, W.rowranges[rownode], W.colranges[colnode])

_orient(A, ::Val{:N}) = A
_orient(A, ::Val{:T}) = transpose(A)

function _checkworkspaceblock(W, k)
    csr = W.csr
    rows, cols = W.rowranges[W.entryrows[k]], W.colranges[csr.colnodes[k]]
    block = csr.isfar[k] ? W.far[csr.storageidx[k]] : W.near[csr.storageidx[k]]
    size(block) == (length(rows), length(cols)) ||
        throw(DimensionMismatch("stored block dimensions differ from clusters"))
    _checkproductoperand(block)
    return nothing
end
