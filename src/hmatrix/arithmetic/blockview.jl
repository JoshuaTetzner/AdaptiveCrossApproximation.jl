"""
    HMatrixBlock{T}

Non-owning rectangular block `rows × cols` of a [`_TreeView`](@ref), in tree
order. `entries` lists the CSR entries of the terminal blocks that meet the
block; terminals crossing its border are clipped to it by the operations that
use it. Hierarchical blocks are the operands of the internal products, sums and
triangular solves. Construct them with [`blockview`](@ref).
"""
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
    for child in children(testtree(W.tree), node)
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
    for child in children(trialtree(W.tree), node)
        _plancolentries!(entries, W, child, rows, cols)
    end
    return entries
end

"""
    blockview(W::_TreeView)
    blockview(W::_TreeView, rows::UnitRange, cols::UnitRange)
    blockview(W::_TreeView, rownode::Int, colnode::Int)

Hierarchical block of `W` over tree-order `rows × cols`, or over the ranges of a
row and a column cluster. The terminal entries are collected by descending the
cluster tree of the axis of which the block covers the smaller fraction. The
full view needs no entry vector.
"""
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
            _planrowentries!(entries, W, root(testtree(W.tree)), rows, cols)
        else
            _plancolentries!(entries, W, root(trialtree(W.tree)), rows, cols)
        end
        sort!(entries)
    end
    return HMatrixBlock{T,typeof(W),typeof(entries)}(W, rows, cols, entries)
end

blockview(W::_TreeView, rownode::Int, colnode::Int) =
    blockview(W, W.rowranges[rownode], W.colranges[colnode])

_orient(A, ::Val{:N}) = A
_orient(A, ::Val{:T}) = transpose(A)
