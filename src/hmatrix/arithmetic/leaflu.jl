"""
    LeafLUFailure(node, pivot)

An exactly zero pivot was encountered in a dense diagonal leaf. `node` is the
caller-supplied cluster ID (zero if unspecified); `pivot` is local to the leaf.
Leaf-restricted pivoting cannot repair a singular leaf using rows outside it.
"""
struct LeafLUFailure <: Exception
    node::Int
    pivot::Int
end

function Base.showerror(io::IO, error::LeafLUFailure)
    return print(
        io,
        "dense leaf LU failed at cluster ",
        error.node,
        ", local pivot ",
        error.pivot,
        " (zero pivot; no regularization applied)",
    )
end

"""
    DenseLeafLU <: LinearAlgebra.Factorization
    leaflu!(A; node=0, check=true, checkfinite=true)
    leaflu!(F::DenseLeafLU; check=true, checkfinite=true)

Pivoted LU `P*A = L*U` (unit-diagonal `L`) of a square dense diagonal leaf, in
place: `F.factors === A`. The second form factorizes new values written into
`F.factors`, reusing the pivot storage. `ipiv` holds LAPACK's sequence of local
row swaps. `info == 0` means success, a positive `info` a zero pivot, which
throws a [`LeafLUFailure`](@ref) unless `check=false`; `info == -1` marks an
invalid state. `checkfinite=false` skips the scan for nonfinite input.
"""
mutable struct DenseLeafLU{T,M<:StridedMatrix{T}} <: LinearAlgebra.Factorization{T}
    factors::M
    ipiv::Vector{BlasInt}
    info::BlasInt
    node::Int
end

Base.size(F::DenseLeafLU) = size(F.factors)
Base.size(F::DenseLeafLU, dim::Integer) = size(F.factors, dim)
LinearAlgebra.issuccess(F::DenseLeafLU) = iszero(F.info)

# Julia 1.10 has no getrf! method accepting caller-owned pivots.
@static if hasmethod(LAPACK.getrf!, Tuple{Matrix{Float64},Vector{BlasInt}})
    _leafgetrf!(A, ipiv) = LAPACK.getrf!(A, ipiv; check=false)[3]
else
    for (T, symbol) in (
        (Float32, :sgetrf_),
        (Float64, :dgetrf_),
        (ComplexF32, :cgetrf_),
        (ComplexF64, :zgetrf_),
    )
        @eval function _leafgetrf!(A::StridedMatrix{$T}, ipiv::Vector{BlasInt})
            info = Ref{BlasInt}()
            ccall(
                (@blasfunc($symbol), LinearAlgebra.libblastrampoline),
                Cvoid,
                (
                    Ref{BlasInt},
                    Ref{BlasInt},
                    Ptr{$T},
                    Ref{BlasInt},
                    Ptr{BlasInt},
                    Ref{BlasInt},
                ),
                size(A, 1),
                size(A, 2),
                A,
                max(1, stride(A, 2)),
                ipiv,
                info,
            )
            LAPACK.chkargsok(info[])
            return info[]
        end
    end
end

function _checkleafstorage(F)
    A = F.factors
    Base.require_one_based_indexing(A, F.ipiv)
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("diagonal leaf must be square"))
    # Empty matrices can have a zero column stride; never pass them to BLAS.
    isempty(A) || _lapackmatrix(A)
    length(F.ipiv) == size(A, 1) || throw(DimensionMismatch("invalid leaf pivot length"))
    return nothing
end

function _checkleafstate(F)
    F.info >= 0 || throw(ArgumentError("dense leaf factorization is invalid"))
    F.info > 0 && throw(LeafLUFailure(F.node, Int(F.info)))
    return nothing
end

function leaflu!(
    A::StridedMatrix{T}; node::Integer=0, check::Bool=true, checkfinite::Bool=true
) where {T<:LinearAlgebra.BlasFloat}
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("diagonal leaf must be square"))
    F = DenseLeafLU(A, Vector{BlasInt}(undef, size(A, 1)), BlasInt(-1), Int(node))
    return leaflu!(F; check, checkfinite)
end

function leaflu!(
    F::DenseLeafLU{T}; check::Bool=true, checkfinite::Bool=true
) where {T<:LinearAlgebra.BlasFloat}
    F.info = -1
    _checkleafstorage(F)
    checkfinite &&
        !all(isfinite, F.factors) &&
        throw(ArgumentError("dense diagonal leaf contains nonfinite input"))
    F.info = isempty(F.factors) ? BlasInt(0) : _leafgetrf!(F.factors, F.ipiv)
    check && F.info > 0 && throw(LeafLUFailure(F.node, Int(F.info)))
    return F
end

@doc (@doc DenseLeafLU) leaflu!

_leafrhs(B::StridedMatrix, side) = B
_leafrhs(B::LowRankMatrix, side) = side == 'L' ? B.U : B.V

# LAPACK ipiv is a swap sequence: forward row swaps form P, reversed ones P'.
function _leafpivots!(B, pivots, inverse)
    n = length(pivots)
    for i in (inverse ? (n:-1:1) : (1:n))
        p = Int(pivots[i])
        i == p && continue
        for j in axes(B, 2)
            B[i, j], B[p, j] = B[p, j], B[i, j]
        end
    end
    return B
end
