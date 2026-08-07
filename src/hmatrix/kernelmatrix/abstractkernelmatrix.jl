using OhMyThreads

"""
    AbstractKernelMatrix{T}

Abstract matrix-like interface for kernel-based entry evaluation used by ACA-style compressors.

# Arguments

  - `T`: scalar element type returned by kernel evaluations

# Returns

A subtype that supports lazy matrix entry access through the kernel matrix interface.

# Notes

Implement this type when matrix entries are computed on demand from geometric/operator data.

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

Build a BEAST-backed kernel matrix (e.g. [`BEASTKernelMatrix`](@ref) or
[`GPUBEASTKernelMatrix`](@ref)) from a BEAST operator, spaces, and `matrixdata`
(a quadrature strategy, or a [`GPUMatrixData`](@ref) for GPU assembly).

No default method is provided here; it is implemented by the `ACABEAST` and
`ACABEASTCUDA` package extensions and dispatched to from
`AbstractKernelMatrix(operator, testspace, trialspace; matrixdata=...)` when BEAST
types are detected.
"""
function beastkernelmatrix end

"""
    assemble_blocks(matrix::AbstractKernelMatrix, blocks, rowidcs, colidcs; scheduler, pbar) -> blocks

Fill each preallocated `blocks[i]` (sized `(length(rowidcs[i]), length(colidcs[i]))`)
by calling `matrix(blocks[i], rowidcs[i], colidcs[i])` once per `i`.

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

"""
    assemble_blocks_sparse(matrix::AbstractKernelMatrix, blocks, rowidcs, colidcs; kwargs...) -> blocks

Batched counterpart of [`assemble_blocks`](@ref): fills every preallocated
`blocks[i]` in one combined pass instead of one `matrix(block, rowidcs[i],
colidcs[i])` call per `i`. Same `rowidcs`/`colidcs`/`blocks` contract as
`assemble_blocks`.

Default (any `AbstractKernelMatrix`) implementation: just forwards to
`assemble_blocks` (no batching benefit without a backend-specific override of
`assemble_blocks` itself). For [`GPUBEASTKernelMatrix`](@ref) (`ACABEASTCUDA`
extension), `assemble_blocks` is itself overridden to batch everything
through one combined pass — see BEAST's `gpu_batched_blockassemble!`, which
it wraps — so this generic method picks that up automatically without its
own override. Works with either quadrature strategy the matrix was built
with; a plain `BEAST.DoubleNumQStrat` matrix is only valid here for blocks
guaranteed geometrically far (no singularity check at all in that case), same
contract as evaluating such a matrix via the plain call operator.
"""
function assemble_blocks_sparse(matrix::AbstractKernelMatrix, blocks, rowidcs, colidcs; kwargs...)
    return assemble_blocks(matrix, blocks, rowidcs, colidcs; kwargs...)
end

function (M::AbstractKernelMatrix)(_, _, _)
    return throw(ArgumentError("callable is not implemented for $(typeof(M))."))
end

Base.eltype(::AbstractKernelMatrix{T}) where {T} = T

function _kernelmatrix_size(ntest::Int, ntrial::Int, dim)
    dim === nothing && return (ntest, ntrial)
    dim == 1 && return ntest
    dim == 2 && return ntrial
    return error("dim must be either 1 or 2")
end
