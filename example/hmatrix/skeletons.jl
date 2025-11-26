using LinearMaps

struct LowRankMatrix{F} <: LinearMaps.LinearMap{F}
    U::Matrix{F}
    V::Matrix{F}
    z::Vector{F}
end

function LowRankMatrix(U::T, V::T) where {F,T<:AbstractMatrix{F}}
    @assert size(V, 1) == size(U, 2)
    return LowRankMatrix{F}(U, V, zeros(F, size(U, 2)))
end

Base.size(lrm::LowRankMatrix) = (size(lrm.U, 1), size(lrm.V, 2))

function LinearAlgebra.mul!(y::AbstractVecOrMat, M::LowRankMatrix, x::AbstractVector)
    mul!(M.z, M.V, x)
    return mul!(y, M.U, M.z)
end

function LinearAlgebra.mul!(
    y::AbstractVecOrMat, M::LinearMaps.TransposeMap{F,T}, x::AbstractVector
) where {F,T<:LowRankMatrix{F}}
    mul!(M.lmap.z, transpose(M.lmap.U), x)
    return mul!(y, transpose(M.lmap.V), M.lmap.z)
end

function LinearAlgebra.mul!(
    y::AbstractVecOrMat, M::LinearMaps.AdjointMap{F,T}, x::AbstractVector
) where {F,T<:LowRankMatrix{F}}
    mul!(M.lmap.z, adjoint(M.lmap.U), x)
    return mul!(y, adjoint(M.lmap.V), M.lmap.z)
end
struct MatrixBlock{I,F,T}
    M::T
    τ::Vector{I}
    σ::Vector{I}
end

function MatrixBlock(M::T, τ::Vector{I}, σ::Vector{I}) where {I,F,T<:AbstractMatrix{F}}
    return MatrixBlock{I,F,T}(M, τ, σ)
end

Base.eltype(block::MatrixBlock{I,F,T}) where {I,F,T} = F
Base.size(block::MatrixBlock) = (length(block.τ), length(block.σ))

LinearAlgebra.rank(block::MatrixBlock) = size(block.M, 2)

function nnzs(lmrb::MatrixBlock{I,F,T}) where {I,F,T<:LowRankMatrix{F}}
    return size(lmrb.M.U, 1) * size(lmrb.M.U, 2) + size(lmrb.M.V, 1) * size(lmrb.M.V, 2)
end
