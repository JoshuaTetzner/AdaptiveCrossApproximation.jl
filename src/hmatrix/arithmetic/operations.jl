# Scaling of H-matrices and helpers shared by the hierarchical arithmetic. Every
# operation recompresses to the largest tolerance of the matrices involved.

_options(matrices...) =
    (; atol=0, rtol=maximum(H.tol for H in matrices), maxrank=typemax(Int))

function _withtol(H::HMatrixCore{T}, tol) where {T}
    return HMatrix{T}(H.tree, H.near, H.far, H.csr, H.csc, H.ordering, H.dim, tol)
end

function _checkeltypes(matrices...)
    T = eltype(first(matrices))
    all(H -> eltype(H) == T, matrices) ||
        throw(ArgumentError("H-matrix arithmetic requires equal element types"))
    return T
end

"""
    rmul!(H::HMatrixCore, α)
    lmul!(α, H::HMatrixCore)

Scale `H` in place by `α`. Exact; dense blocks are scaled and every low-rank
block scales its smaller factor. `α = 0` leaves rank-zero low-rank blocks.
"""
function LinearAlgebra.rmul!(H::HMatrixCore{T}, alpha::Number) where {T}
    alpha = convert(T, alpha)
    isone(alpha) && return H
    for D in H.near
        iszero(alpha) ? fill!(D, zero(T)) : rmul!(D, alpha)
    end
    foreach(R -> _scalelowrank!(R, alpha), H.far)
    return H
end
LinearAlgebra.lmul!(alpha::Number, H::HMatrixCore) = rmul!(H, alpha)

# Union{Real,Complex} rather than Number: LinearMaps defines lazy scaled maps
# for these scalars.
Base.:*(alpha::Union{Real,Complex}, H::HMatrixCore) = rmul!(copy(H), alpha)
Base.:*(H::HMatrixCore, alpha::Union{Real,Complex}) = rmul!(copy(H), alpha)
Base.:-(H::HMatrixCore) = rmul!(copy(H), -1)
