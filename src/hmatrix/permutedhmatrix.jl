
struct PermutedHMatrix{K,HMatrixType} <: LinearMaps.LinearMap{K}
    permutation::Tuple{Vector{Int},Vector{Int}}
    M::HMatrixType
    function PermutedHMatrix{K}(permutation, hmatrix) where {K}
        return new{K,typeof(hmatrix)}(permutation, hmatrix)
    end
end

function Base.size(A::PermutedHMatrix, dim=nothing)
    dim === nothing && return (A.dim[1], A.dim[2])
    return A.dim[dim]
end

function LinearMaps._unsafe_mul!(
    y::AbstractVector, A::M, x::AbstractVector
) where {K,M<:PermutedHMatrix{K}}
    return LinearMaps._unsafe_mul!(y[A.permutation[1]], A.M, x[A.permutation[2]])
end

function LinearMaps._unsafe_mul!(
    y::AbstractVector, A::M, x::AbstractVector
) where {K,Z<:PermutedHMatrix{K},M<:LinearMaps.TransposeMap{<:Any,Z}}
    return LinearMaps._unsafe_mul!(
        y[A.lmap.permutation[1]], transpose(A.lmap.M), x[A.lmap.permutation[2]]
    )
end

function LinearMaps._unsafe_mul!(
    y::AbstractVector, A::M, x::AbstractVector
) where {K,Z<:PermutedHMatrix{K},M<:LinearMaps.AdjointMap{<:Any,Z}}
    return LinearMaps._unsafe_mul!(
        y[A.lmap.permutation[1]], adjoint(A.lmap.M), x[A.lmap.permutation[2]]
    )
end
