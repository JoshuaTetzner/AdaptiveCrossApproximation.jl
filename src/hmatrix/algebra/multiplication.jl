# Factor views of a low-rank block restricted to some of its rows and columns.
struct _LowRankSlice{T,U<:AbstractMatrix{T},V<:AbstractMatrix{T}} <: LinearMaps.LinearMap{T}
    U::U
    V::V
end
Base.size(A::_LowRankSlice) = (size(A.U, 1), size(A.V, 2))

const _LowRankOperand{T} = Union{LowRankMatrix{T},_LowRankSlice{T}}

_productfactors(A::LowRankMatrix) = (A.U, A.V)
_productfactors(A::_LowRankSlice) = (A.U, A.V)

_blockarrays(A::AbstractMatrix) = (A,)
_blockarrays(A::_LowRankOperand) = _productfactors(A)

function _checkproductoperand(A::AbstractMatrix)
    Base.require_one_based_indexing(A)
    return nothing
end
function _checkproductoperand(A::_LowRankOperand)
    U, V = _productfactors(A)
    size(U, 2) == size(V, 1) || throw(DimensionMismatch("incompatible low-rank factors"))
    return nothing
end

function _checkproductalias(C, A)
    for factor in _blockarrays(A)
        Base.mightalias(C, factor) &&
            throw(ArgumentError("product outputs must not overlap inputs or each other"))
    end
    return nothing
end
