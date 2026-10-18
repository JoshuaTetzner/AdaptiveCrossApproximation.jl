"""
    LowRankMatrix{T}

Low-rank numerical block represented by factors `U` and `V`. Temporary
multiplication storage deliberately does not belong to the block.
The block is mutable so arithmetic can replace its factors when its rank changes.
"""
mutable struct LowRankMatrix{T} <: LinearMaps.LinearMap{T}
    U::Matrix{T}
    V::Matrix{T}
    function LowRankMatrix{T}(U::Matrix{T}, V::Matrix{T}) where {T}
        @assert size(V, 1) == size(U, 2)
        return new{T}(U, V)
    end
end

function LowRankMatrix(U::AbstractMatrix{T}, V::AbstractMatrix{T}) where {T}
    @assert size(V, 1) == size(U, 2)
    return LowRankMatrix{T}(Matrix(U), Matrix(V))
end

Base.eltype(::Type{<:LowRankMatrix{T}}) where {T} = T
Base.size(A::LowRankMatrix) = (size(A.U, 1), size(A.V, 2))
Base.Matrix(A::LowRankMatrix) = A.U * A.V

LinearAlgebra.rank(A::LowRankMatrix) = size(A.U, 2)
_storedfarblockentries(A::LowRankMatrix) = length(A.U) + length(A.V)

# Allocating products of single blocks; H-matrix products use MatVecWorkspace.
function LinearAlgebra.mul!(
    y::AbstractVecOrMat, A::LowRankMatrix, x::AbstractVector, alpha::Number, beta::Number
)
    z = A.V * x
    return LinearAlgebra.mul!(y, A.U, z, alpha, beta)
end

function LinearAlgebra.mul!(y::AbstractVecOrMat, A::LowRankMatrix, x::AbstractVector)
    return LinearAlgebra.mul!(y, A, x, true, false)
end

function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    A::LinearMaps.TransposeMap{<:Any,<:LowRankMatrix},
    x::AbstractVector,
    alpha::Number,
    beta::Number,
)
    z = transpose(A.lmap.U) * x
    return LinearAlgebra.mul!(y, transpose(A.lmap.V), z, alpha, beta)
end

function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    A::LinearMaps.TransposeMap{<:Any,<:LowRankMatrix},
    x::AbstractVector,
)
    return LinearAlgebra.mul!(y, A, x, true, false)
end

function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    A::LinearMaps.AdjointMap{<:Any,<:LowRankMatrix},
    x::AbstractVector,
    alpha::Number,
    beta::Number,
)
    z = adjoint(A.lmap.U) * x
    return LinearAlgebra.mul!(y, adjoint(A.lmap.V), z, alpha, beta)
end

function LinearAlgebra.mul!(
    y::AbstractVecOrMat, A::LinearMaps.AdjointMap{<:Any,<:LowRankMatrix}, x::AbstractVector
)
    return LinearAlgebra.mul!(y, A, x, true, false)
end
