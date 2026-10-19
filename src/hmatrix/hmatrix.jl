"""
    HMatrixCore{T}

Hierarchical matrix with a retained block tree, separate dense and low-rank
storage, canonical CSR block ordering, and a CSC traversal map.

`tol` is the relative accuracy chosen at assembly. Every later recompression
(products, sums, [`recompress!`](@ref) and `lu`) truncates each block to this
tolerance; for operands with different tolerances the larger one is used.

The struct is immutable and operations never change the blocks of their
inputs. Only the documented in-place operations overwrite a matrix: the
destination of `mul!(C, A, B, α, β)`, `axpy!`, `axpby!`, `rmul!` and `lmul!`, and
the argument of `lu!` and `recompress!`.
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

"""
    copy(H::HMatrixCore)

Copy the numerical blocks of `H`. Tree and block index are shared, so the copy
has the same partition and can reuse workspaces built for `H`.
"""
function Base.copy(A::HMatrix{T}) where {T}
    return HMatrix{T}(
        A.tree,
        map(copy, A.near),
        [LowRankMatrix{T}(copy(R.U), copy(R.V)) for R in A.far],
        A.csr,
        A.csc,
        A.ordering,
        A.dim,
        A.tol,
    )
end

function _materialize(A::HMatrix, farfilter::Union{Nothing,Bool}=nothing)
    matrix = zeros(eltype(A), size(A))
    rowtree = testtree(A.tree)
    coltree = trialtree(A.tree)
    for rownode in 1:(length(A.csr.rowptr) - 1)
        rows = values(rowtree, rownode)
        for csridx in A.csr.rowptr[rownode]:(A.csr.rowptr[rownode + 1] - 1)
            isnothing(farfilter) || A.csr.isfar[csridx] == farfilter || continue
            cols = values(coltree, A.csr.colnodes[csridx])
            k = A.csr.storageidx[csridx]
            block = Matrix(A.csr.isfar[csridx] ? A.far[k] : A.near[k])
            view(matrix, rows, cols) .= A.csr.transposed[csridx] ? transpose(block) : block
        end
    end
    return matrix
end

Base.Matrix(A::HMatrix) = _materialize(A)

"""
    nnz(H::HMatrixCore)

Number of stored scalars of `H`: the entries of the dense blocks plus
`length(U) + length(V)` for every low-rank block. Blocks shared by both
orientations of a symmetric matrix count once. Index arrays and object overhead
are not included. See also [`storage`](@ref).
"""
function nnz(A::HMatrix)
    return sum(length, A.near; init=0) + sum(_storedfarblockentries, A.far; init=0)
end

"""
    HMatrixComponent{T,H,Far}

Lazy near-field (`Far = false`) or far-field (`Far = true`) part of an
H-matrix, returned by [`nearmatrix`](@ref) and [`farmatrix`](@ref). It shares
the blocks of its parent and supports `Matrix` and products, also through a
[`MatVecWorkspace`](@ref).
"""
struct HMatrixComponent{T,H<:HMatrix{T},Far} <: LinearMaps.LinearMap{T}
    parent::H
end

Base.size(A::HMatrixComponent) = size(A.parent)
Base.size(A::HMatrixComponent, dim::Integer) = size(A.parent, dim)
Base.Matrix(A::HMatrixComponent{<:Any,<:Any,Far}) where {Far} = _materialize(A.parent, Far)

"""
    nearmatrix(H::HMatrixCore)

Near-field part of `H`: the dense blocks only, as a lazy linear map sharing the
blocks of `H` (see [`HMatrixComponent`](@ref)). Useful to analyse the near-field
structure or to apply it separately.
"""
nearmatrix(A::HMatrix{T}) where {T} = HMatrixComponent{T,typeof(A),false}(A)

"""
    farmatrix(H::HMatrixCore)

Far-field part of `H`: the low-rank blocks only, as a lazy linear map sharing the
blocks of `H` (see [`HMatrixComponent`](@ref)). Useful to analyse the rank
structure or to apply it separately.
"""
farmatrix(A::HMatrix{T}) where {T} = HMatrixComponent{T,typeof(A),true}(A)

"""
    storage(H::HMatrixCore)

Report the storage of `H`. Prints the numerical block payload ([`nnz`](@ref)
scalars) in GB, the complete Julia object footprint reported by
`Base.summarysize` in GB, and the ratio of the payload to dense storage.

# Returns

  - `Float64`: numerical block payload in GB
"""
function storage(A::HMatrix)
    entries = nnz(A)
    bytes = entries * sizeof(eltype(A))
    referencebytes = prod(size(A)) * sizeof(eltype(A))
    println("storage: ", bytes * 1e-9, " GB")
    println("summary size: ", Base.summarysize(A) * 1e-9, " GB")
    println("compression ratio: ", bytes / referencebytes)
    return bytes * 1e-9
end
