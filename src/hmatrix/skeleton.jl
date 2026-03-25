struct LowRankMatrix{K} <: LinearMaps.LinearMap{K}
    U::Matrix{K}
    V::Matrix{K}
    z::Vector{K}
end

function LowRankMatrix(U::T, V::T) where {K,T<:AbstractMatrix{K}}
    @assert size(V, 1) == size(U, 2)
    return LowRankMatrix{K}(U, V, zeros(K, size(U, 2)))
end

Base.eltype(::Type{<:LowRankMatrix{K}}) where {K} = K
Base.size(lrm::LowRankMatrix) = (size(lrm.U, 1), size(lrm.V, 2))

function LinearAlgebra.mul!(
    y::AbstractVecOrMat, M::LowRankMatrix{T}, x::AbstractVector
) where {T}
    LinearAlgebra.mul!(M.z, M.V, x)
    return LinearAlgebra.mul!(y, M.U, M.z)
end

function LinearAlgebra.mul!(
    y::AbstractVecOrMat, M::LinearMaps.TransposeMap{T,K}, x::AbstractVector
) where {T,K<:LowRankMatrix{T}}
    LinearAlgebra.mul!(M.lmap.z, transpose(M.lmap.U), x)
    return LinearAlgebra.mul!(y, transpose(M.lmap.V), M.lmap.z)
end

function LinearAlgebra.mul!(
    y::AbstractVecOrMat, M::LinearMaps.AdjointMap{T,K}, x::AbstractVector
) where {T,K<:LowRankMatrix{T}}
    LinearAlgebra.mul!(M.lmap.z, adjoint(M.lmap.U), x)
    return LinearAlgebra.mul!(y, adjoint(M.lmap.V), M.lmap.z)
end
