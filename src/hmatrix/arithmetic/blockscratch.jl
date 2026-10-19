using LinearAlgebra: BlasInt
using LinearAlgebra.BLAS: @blasfunc
using Base.Checked: checked_add, checked_mul

"""
    BlockScratch{T}()

Reusable temporary storage of one task for low-rank block arithmetic. The
buffers start empty, grow on demand and never shrink; their layout is private to
the kernels. Final numerical blocks never reference scratch storage.
"""
mutable struct BlockScratch{T}
    left::Vector{T}
    right::Vector{T}
    small::Vector{T}
    tauleft::Vector{T}
    tauright::Vector{T}
    productleft::Vector{T}
    productright::Vector{T}
    aggregate::Vector{T}
end

BlockScratch{T}() where {T} = BlockScratch{T}(T[], T[], T[], T[], T[], T[], T[], T[])

"""
    _ProductStatus

Truncation status accumulated over a sequence of recompressions: whether all
met the tolerance (`converged`), whether any was capped by `maxrank`
(`rank_limited`), the largest discarded spectral error of a block
(`max_block_error`) and the number of recompressions (`flushes`). Updated with
`_record!`; reported by `lu!` through [`HMatrixLU`](@ref).
"""
mutable struct _ProductStatus{R}
    converged::Bool
    rank_limited::Bool
    max_block_error::R
    flushes::Int
end

function _record!(status::_ProductStatus, info, flushes=1)
    status.converged &= info.converged
    status.rank_limited |= info.rank_limited
    status.max_block_error = max(status.max_block_error, info.error)
    status.flushes += flushes
    return status
end

function _lapackmatrix(A)
    Base.require_one_based_indexing(A)
    stride(A, 1) == 1 && stride(A, 2) >= max(1, size(A, 1)) ||
        throw(ArgumentError("LAPACK requires column-major storage with unit row stride"))
    return nothing
end

function _growscratch!(buffer::Vector, required::Int)
    required > length(buffer) && resize!(buffer, max(required, 2 * length(buffer)))
    return buffer
end
