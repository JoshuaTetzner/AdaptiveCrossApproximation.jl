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
function AbstractKernelMatrix(operator, testspace, trialspace; args...) end

function (::AbstractKernelMatrix)(matrixblock, tdata, sdata) end

Base.eltype(::AbstractKernelMatrix{T}) where {T} = T
