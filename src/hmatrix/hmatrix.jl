
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
`HMatrix(operator, testspace, trialspace, tree; kwargs...)` or `assemble(...)`.

# See also

`HMatrix`, `assemble`, `farmatrix`, `nearmatrix`
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

"""
    nnz(A::HMatrix)

Count the number of stored scalars in a hierarchical matrix.

Sums the near-field block-sparse storage with the far-field low-rank factor
storage (`length(U) + length(V)` per block), i.e. the actual memory footprint
in matrix entries rather than the dense `size(A,1) * size(A,2)`. See also
[`storage`](@ref) for a GB-scale report including compression ratio.
"""
function nnz(A::HMatrix)
    farnnz = sum(
        length(blk.U) + length(blk.V) for f in A.farinteractions for blk in f.blocks; init=0
    )
    return BlockSparseMatrices.nnz(A.nearinteractions) + farnnz
end

function Base.size(A::HMatrix, dim=nothing)
    dim === nothing && return (A.dim[1], A.dim[2])
    return A.dim[dim]
end

function LinearMaps._unsafe_mul!(
    y::AbstractVector, A::HMatrix{K}, x::AbstractVector
) where {K}
    fill!(y, zero(K))

    y .+= A.nearinteractions * x
    for farinteraction in A.farinteractions
        y .+= farinteraction * x
    end

    return y
end

function LinearMaps._unsafe_mul!(
    y::AbstractVector, A::LinearMaps.TransposeMap{<:Any,<:HMatrix{K}}, x::AbstractVector
) where {K}
    fill!(y, zero(K))

    y .+= transpose(A.lmap.nearinteractions) * x
    for farinteraction in A.lmap.farinteractions
        y .+= transpose(farinteraction) * x
    end

    return y
end

function LinearMaps._unsafe_mul!(
    y::AbstractVector, A::LinearMaps.AdjointMap{<:Any,<:HMatrix{K}}, x::AbstractVector
) where {K}
    fill!(y, zero(K))

    y .+= adjoint(A.lmap.nearinteractions) * x
    for farinteraction in A.lmap.farinteractions
        y .+= adjoint(farinteraction) * x
    end

    return y
end
