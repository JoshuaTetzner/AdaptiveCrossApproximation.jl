
struct HMatrix{K,NearInteractionType,FarInteractionType} <: LinearMaps.LinearMap{K}
    nearinteractions::NearInteractionType
    farinteractions::FarInteractionType
    dim::Tuple{Int,Int}
    function HMatrix{K}(nearinteractions, farinteractions, dim) where {K}
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

function storage(A::HMatrix)
    storage = BlockSparseMatrices.nnz(A.nearinteractions) * 8 * 1e-9
    println("Nearinteractions: $storage MB")
    fstorage = 0
    for farinteraction in A.farinteractions
        fstorage += BlockSparseMatrices.nnz(farinteraction)
    end
    fstorage = fstorage * 8 * 1e-9
    println("Farinteractions: $fstorage MB")
    return storage + fstorage
end

function Base.size(A::HMatrix, dim=nothing)
    dim === nothing && return (A.dim[1], A.dim[2])
    return A.dim[dim]
end

function LinearMaps._unsafe_mul!(
    y::AbstractVector, A::M, x::AbstractVector
) where {K,M<:HMatrix{K}}
    fill!(y, zero(K))

    mul!(y, A.nearinteractions, x)
    for farinteraction in A.farinteractions
        y += farinteraction * x
    end

    return y
end

function LinearMaps._unsafe_mul!(
    y::AbstractVector, A::M, x::AbstractVector
) where {K,Z<:HMatrix{K},M<:LinearMaps.TransposeMap{<:Any,Z}}
    fill!(y, zero(K))

    mul!(y, transpose(A.lmap.nearinteractions), x)
    for farinteraction in A.lmap.farinteractions
        y += transpose(farinteraction) * x
    end

    return y
end

function LinearMaps._unsafe_mul!(
    y::AbstractVector, A::M, x::AbstractVector
) where {K,Z<:HMatrix{K},M<:LinearMaps.AdjointMap{<:Any,Z}}
    fill!(y, zero(K))

    mul!(y, adjoint(A.lmap.nearinteractions), x)
    for farinteraction in A.lmap.farinteractions
        y += adjoint(farinteraction) * x
    end

    return y
end
