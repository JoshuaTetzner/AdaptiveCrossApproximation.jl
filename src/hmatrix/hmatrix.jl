
"""
    HMatrix{K,NearInteractionType,FarInteractionType}

Hierarchical matrix that stores near-field interactions explicitly and far-field interactions
as low-rank block data.

# Arguments
- `nearinteractions`: block-sparse near-field contribution
- `farinteractions`: collection of compressed far-field interaction blocks
- `dim::Tuple{Int,Int}`: matrix dimensions `(m, n)`

# Returns
An `HMatrix` linear map that supports matrix-vector products and conversion to a dense matrix
via `Matrix`.

# Notes
`HMatrix` is typically created through the high-level constructor
`HMatrix(operator, testspace, trialspace, tree; kwargs...)` or `H.assemble(...)`.

# See also
`HMatrix`, `H.assemble`, `farmatrix`, `nearmatrix`
"""
struct HMatrix{K,NearInteractionType,FarInteractionType} <: LinearMaps.LinearMap{K}
    nearinteractions::NearInteractionType
    farinteractions::FarInteractionType
    dim::Tuple{Int,Int}
    function HMatrix{K}(nearinteractions, farinteractions, dim::Tuple{Int,Int}) where {K}
        return new{K,typeof(nearinteractions),typeof(farinteractions)}(
            nearinteractions, farinteractions, dim
        )
    end
end

function Base.Matrix(A::HMatrix)
    mat = Matrix(A.nearinteractions)
    for farinteraction in A.farinteractions
        mat += Matrix(farinteraction)
    end
    return mat
end

function nnz(A::HMatrix)
    nnz = BlockSparseMatrices.nnz(A.nearinteractions)
    println("Nearinteractions: $nnz")
    fnnz = 0
    for farinteraction in A.farinteractions
        fnnz += BlockSparseMatrices.nnz(farinteraction)
    end
    println("Farinteractions: $fnnz")
    return nnz + fnnz
end

function Base.size(A::HMatrix, dim=nothing)
    dim === nothing && return (A.dim[1], A.dim[2])
    return A.dim[dim]
end

function LinearMaps._unsafe_mul!(
    y::AbstractVector, A::M, x::AbstractVector
) where {K,M<:HMatrix{K}}
    fill!(y, zero(K))

    y .+= A.nearinteractions * x
    for farinteraction in A.farinteractions
        y .+= farinteraction * x
    end

    return y
end

function LinearMaps._unsafe_mul!(
    y::AbstractVector, A::M, x::AbstractVector
) where {K,Z<:HMatrix{K},M<:LinearMaps.TransposeMap{<:Any,Z}}
    fill!(y, zero(K))

    y .+= transpose(A.lmap.nearinteractions) * x
    for farinteraction in A.lmap.farinteractions
        y .+= transpose(farinteraction) * x
    end

    return y
end

function LinearMaps._unsafe_mul!(
    y::AbstractVector, A::M, x::AbstractVector
) where {K,Z<:HMatrix{K},M<:LinearMaps.AdjointMap{<:Any,Z}}
    fill!(y, zero(K))

    y .+= adjoint(A.lmap.nearinteractions) * x
    for farinteraction in A.lmap.farinteractions
        y .+= adjoint(farinteraction) * x
    end

    return y
end
