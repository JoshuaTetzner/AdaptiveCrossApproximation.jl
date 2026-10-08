using OhMyThreads

"""
    AbstractKernelMatrix{T}

Abstract matrix-like interface for kernel-based entry evaluation used by ACA-style compressors.

# Arguments

  - `T`: scalar element type returned by kernel evaluations

# Returns

A subtype that supports lazy matrix entry access through the kernel matrix interface.

# Notes

Implement this type when matrix entries are computed on demand from
geometric/operator data. A concrete backend should implement `size(matrix)` and
`matrix(block, rows, columns)`; [`nextrc!`](@ref) uses that callable interface to
sample rows and columns without materializing the full matrix.

# See also

`AbstractKernelMatrix(operator, testspace, trialspace; args...)`
"""
abstract type AbstractKernelMatrix{T} end

"""
    AbstractKernelMatrix(operator, testspace, trialspace; args...)

Construct a concrete kernel matrix wrapper from operator and space data.

# Arguments

  - `operator`: operator or kernel definition
  - `testspace`: space for row evaluation points or basis data
  - `trialspace`: space for column evaluation points or basis data
  - `args...`: backend-specific keyword arguments

# Returns

A concrete subtype of `AbstractKernelMatrix` provided by method dispatch.

# Notes

This declaration defines the interface entry point. Concrete backends provide
specialized methods for specific operator/space types.

# See also

`AbstractKernelMatrix`, `nextrc!`
"""
function AbstractKernelMatrix(operator, testspace, trialspace; args...)
    return error(
        "AbstractKernelMatrix is not implemented for operator::$(typeof(operator)), " *
        "testspace::$(typeof(testspace)), trialspace::$(typeof(trialspace)).",
    )
end

"""
    beastkernelmatrix(operator, testspace, trialspace, matrixdata)

Build a BEAST-backed kernel matrix (e.g. [`BEASTKernelMatrix`](@ref)) from a
BEAST operator, spaces, and `matrixdata` (a quadrature strategy).

No default method is provided here; it is implemented by the `ACABEAST`
package extension and dispatched to from
`AbstractKernelMatrix(operator, testspace, trialspace; matrixdata=...)` when BEAST
types are detected.
"""
function beastkernelmatrix end

"""
    assemble_blocks(matrix::AbstractKernelMatrix, blocks, rowidcs, colidcs; scheduler, pbar) -> blocks

Fill each preallocated `blocks[i]` (sized `(length(rowidcs[i]), length(colidcs[i]))`)
by calling `matrix(blocks[i], rowidcs[i], colidcs[i])` once per `i`.
The built-in kernel backends accumulate into the supplied blocks, so callers should
normally provide zero-initialized storage.

`rowidcs`/`colidcs` need not describe disjoint or contiguous index ranges —
this is the same shape of input as [`nearinteractions`](@ref) (dense near
blocks of a fast method) but also fits scattered pivot-index blocks (e.g. the
far-field coupling blocks of a nested cross approximation), since both are
just lists of (row-index-list, column-index-list) requests against the same
underlying kernel.

Taking `blocks` as an argument (rather than allocating and returning it) lets
callers own the storage layout (e.g. a `BlockSparseMatrix`'s block vector) and
lets backends override just the fill strategy — e.g. a GPU backend can force a
serial loop, or batch everything through one combined pass — without
duplicating the surrounding preallocation/storage-construction code.

Default (any `AbstractKernelMatrix`) implementation: a plain per-block loop,
`@tasks`-parallelizable via `scheduler`.
"""
function assemble_blocks(
    matrix::AbstractKernelMatrix{T},
    blocks::AbstractVector{<:AbstractMatrix{T}},
    rowidcs::AbstractVector{<:AbstractVector{Int}},
    colidcs::AbstractVector{<:AbstractVector{Int}};
    scheduler=SerialScheduler(),
    pbar=Progress(0; enabled=false),
) where {T}
    @tasks for i in eachindex(blocks)
        @set scheduler = scheduler
        matrix(blocks[i], rowidcs[i], colidcs[i])
        next!(pbar)
    end
    return blocks
end

function (M::AbstractKernelMatrix)(_, _, _)
    return throw(ArgumentError("callable is not implemented for $(typeof(M))."))
end

Base.eltype(::AbstractKernelMatrix{T}) where {T} = T

"""
    localkernelmatrix(matrix::AbstractKernelMatrix, rows, columns)

Prepare a matrix-like object for repeated row and column sampling of one block.

The fallback returns `matrix` unchanged. A backend may return a lightweight local
wrapper that precomputes block-dependent assembly data. The returned object must
accept the original global indices through the same `nextrc!` interface and report
the same global `size` and `eltype` as `matrix`.

This hook is called once per far-field block before ACA compression, so any cached
state belongs to that block and must not be shared concurrently between blocks.
"""
localkernelmatrix(matrix::AbstractKernelMatrix, rows, columns) = matrix

function _kernelmatrix_size(ntest::Int, ntrial::Int, dim)
    dim === nothing && return (ntest, ntrial)
    dim == 1 && return ntest
    dim == 2 && return ntrial
    return error("dim must be either 1 or 2")
end
