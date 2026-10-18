using LinearAlgebra: BlasInt
using LinearAlgebra.BLAS: @blasfunc
using Base.Checked: checked_mul

"""
    BlockScratch{T}()

Reusable, exclusively owned temporary storage for low-rank block arithmetic.
Starts with empty buffers; kernels reshape them as needed, so callers must not
depend on the backing-buffer layout or assume fixed buffer shapes.
Householder coefficients have separate reusable vectors. The reduced-matrix
buffer holds up to `r^2` entries. Newly allocated entries are
uninitialized. Buffers grow geometrically and never shrink
automatically. Final numerical blocks must own storage independent of scratch.
The hierarchical product keeps its pending factors, dense cores and node
aggregates in three further buffers that grow to exact sizes.

One task may use a scratch object at a time.
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

# Every scratch array, for alias checks.
_scratcharrays(scratch::BlockScratch) = (
    scratch.left,
    scratch.right,
    scratch.small,
    scratch.tauleft,
    scratch.tauright,
    scratch.productleft,
    scratch.productright,
    scratch.aggregate,
)

# Aggregated local truncation status of a sequence of recompressions.
mutable struct _ProductStatus{R}
    converged::Bool
    rank_limited::Bool
    max_block_error::R
    flushes::Int
end

function _lapackmatrix(A)
    Base.require_one_based_indexing(A)
    stride(A, 1) == 1 && stride(A, 2) >= max(1, size(A, 1)) ||
        throw(ArgumentError("LAPACK requires column-major storage with unit row stride"))
    return nothing
end

function _blockscratchsizes(m::Integer, n::Integer, r::Integer)
    m >= 0 && n >= 0 && r >= 0 ||
        throw(ArgumentError("block dimensions and temporary rank must be nonnegative"))
    m, n, r = Int(m), Int(n), Int(r)
    left = Base.Checked.checked_mul(m, r)
    right = Base.Checked.checked_mul(n, r)
    small = Base.Checked.checked_mul(min(m, r), min(n, r))
    return m, n, r, left, right, small
end

function _growscratch!(buffer::Vector, required::Int)
    capacity = length(buffer)
    if required > capacity
        doubled = capacity <= typemax(Int) ÷ 2 ? 2capacity : required
        resize!(buffer, max(required, doubled))
    end
    return buffer
end
