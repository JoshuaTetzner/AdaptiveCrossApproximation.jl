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
