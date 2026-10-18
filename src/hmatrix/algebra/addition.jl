# Exact scaling of the smaller factor; zero leaves a rank-zero block.
function _scalelowrank!(R::LowRankMatrix{T}, alpha) where {T}
    m, n = size(R)
    if iszero(alpha)
        R.U, R.V = Matrix{T}(undef, m, 0), Matrix{T}(undef, 0, n)
    else
        m <= n ? rmul!(R.U, alpha) : rmul!(R.V, alpha)
    end
    return R
end

# Replace C by a recompressed approximation of C + alpha*A, with tolerances
# relative to the sum. Source factors are never overwritten. Returns the
# truncation information of `_recompress!`.
function _addlowrank!(
    C::LowRankMatrix{T},
    A::_LowRankOperand{T},
    alpha::Number,
    scratch::BlockScratch{T};
    atol::Real=0,
    rtol::Real=1e-6,
    maxrank::Integer=typemax(Int),
) where {T<:LinearAlgebra.BlasFloat}
    _recompressionoptions(atol, rtol, maxrank)
    size(C) == size(A) || throw(DimensionMismatch("low-rank block dimensions differ"))
    C === A && throw(ArgumentError("C and A must be different blocks"))
    _checkproductoperand(C)
    _checkproductoperand(A)
    alpha = convert(T, alpha)
    isfinite(alpha) || throw(ArgumentError("alpha must be finite"))
    X, Y = _productfactors(A)
    if iszero(alpha) || min(size(X, 2), size(C)...) == 0
        return _recompress!(C, scratch; atol, rtol, maxrank)
    end
    return _flushfactors!(C, X, Y, scratch; atol, rtol, maxrank, alpha)
end
