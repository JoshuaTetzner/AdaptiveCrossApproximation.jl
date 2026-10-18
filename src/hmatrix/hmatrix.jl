"""
    HMatrixCore{T}

Hierarchical matrix with a retained block tree, separate dense and low-rank
storage, canonical CSR block ordering, and a CSC traversal map.

`tol` is the relative accuracy chosen at assembly.
"""
struct HMatrixCore{T,Tree,Ordering} <: LinearMaps.LinearMap{T}
    tree::Tree
    near::Vector{Matrix{T}}
    far::Vector{LowRankMatrix{T}}
    csr::BlockCSR{Int}
    csc::BlockCSC{Int}
    ordering::Ordering
    dim::Tuple{Int,Int}
    tol::Float64
end

"""
    HMatrix

Compatibility alias for [`HMatrixCore`](@ref).
"""
const HMatrix = HMatrixCore

function HMatrix{T}(
    tree,
    near::Vector{Matrix{T}},
    far::Vector{LowRankMatrix{T}},
    csr::BlockCSR{Int},
    csc::BlockCSC{Int},
    ordering,
    dim::Tuple{Int,Int},
    tol::Real,
) where {T}
    isfinite(tol) && tol >= 0 || throw(ArgumentError("tol must be finite and nonnegative"))
    return HMatrix{T,typeof(tree),typeof(ordering)}(
        tree, near, far, csr, csc, ordering, dim, Float64(tol)
    )
end

Base.size(A::HMatrix) = A.dim
Base.size(A::HMatrix, dim::Integer) = A.dim[dim]

_copylowrank(block::LowRankMatrix{T}) where {T} =
    LowRankMatrix{T}(copy(block.U), copy(block.V))

"""
    copy(H::HMatrixCore)

Copy the numerical blocks of `H`. Tree and block index are shared, so the copy
has the same partition and can reuse workspaces built for `H`.
"""
function Base.copy(A::HMatrix{T}) where {T}
    return HMatrix{T}(
        A.tree,
        map(copy, A.near),
        map(_copylowrank, A.far),
        A.csr,
        A.csc,
        A.ordering,
        A.dim,
        A.tol,
    )
end

function _storedblock(A::HMatrix, csridx::Int)
    if A.csr.isfar[csridx]
        return A.far[A.csr.storageidx[csridx]]
    end
    return A.near[A.csr.storageidx[csridx]]
end

function _logicalblock(A::HMatrix, csridx::Int)
    block = _storedblock(A, csridx)
    return A.csr.transposed[csridx] ? transpose(block) : block
end

function _materialize(A::HMatrix, farfilter::Union{Nothing,Bool}=nothing)
    matrix = zeros(eltype(A), size(A))
    rowtree = testtree(A.tree)
    coltree = trialtree(A.tree)
    for rownode in 1:(length(A.csr.rowptr) - 1)
        rows = clusterindices(rowtree, rownode, A.ordering)
        for csridx in A.csr.rowptr[rownode]:(A.csr.rowptr[rownode + 1] - 1)
            isnothing(farfilter) || A.csr.isfar[csridx] == farfilter || continue
            cols = clusterindices(coltree, A.csr.colnodes[csridx], A.ordering)
            view(matrix, rows, cols) .= Matrix(_logicalblock(A, csridx))
        end
    end
    return matrix
end

Base.Matrix(A::HMatrix) = _materialize(A)

function nnz(A::HMatrix)
    return sum(length, A.near; init=0) + sum(_storedfarblockentries, A.far; init=0)
end

struct HMatrixComponent{T,H<:HMatrix{T},Far} <: LinearMaps.LinearMap{T}
    parent::H
end

Base.size(A::HMatrixComponent) = size(A.parent)
Base.size(A::HMatrixComponent, dim::Integer) = size(A.parent, dim)
Base.Matrix(A::HMatrixComponent{<:Any,<:Any,Far}) where {Far} = _materialize(A.parent, Far)

nearmatrix(A::HMatrix{T}) where {T} = HMatrixComponent{T,typeof(A),false}(A)
farmatrix(A::HMatrix{T}) where {T} = HMatrixComponent{T,typeof(A),true}(A)

function storage(A::HMatrix)
    entries = nnz(A)
    bytes = entries * sizeof(eltype(A))
    referencebytes = prod(size(A)) * sizeof(eltype(A))
    println("storage: ", bytes * 1e-9, " GB")
    println("summary size: ", Base.summarysize(A) * 1e-9, " GB")
    println("compression ratio: ", bytes / referencebytes)
    return bytes * 1e-9
end
