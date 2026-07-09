"""
    GPUMatrixData{QuadStratType}

Configuration for GPU-accelerated BEAST kernel matrix assembly.

Passed as the `matrixdata` keyword argument to [`AbstractKernelMatrix`](@ref) to build
a [`GPUBEASTKernelMatrix`](@ref) instead of a plain `BEASTKernelMatrix`; requires the
ACABEASTCUDA package extension to be loaded (BEAST.jl and CUDA.jl available).

# Fields

  - `quadstrat::QuadStratType`: BEAST quadrature strategy used to assemble matrix entries
  - `ndevices::Int`: Requested number of GPU devices (currently must be one)
  - `device::Int`: CUDA device used by the block assembler
"""
struct GPUMatrixData{QuadStratType}
    quadstrat::QuadStratType
    ndevices::Int
    device::Int
    function GPUMatrixData(quadstrat, ndevices::Int=1; device::Int=0)
        ndevices > 0 || throw(ArgumentError("ndevices must be positive"))
        return new{typeof(quadstrat)}(quadstrat, ndevices, device)
    end
end

"""
    GPUBEASTKernelMatrix{T,NearBlockAssemblerType} <: AbstractKernelMatrix{T}

GPU-accelerated kernel matrix wrapper for BEAST operator assembly.

Analogous to [`BEASTKernelMatrix`](@ref), but backed by a GPU near-field block
assembler. Built automatically by `AdaptiveCrossApproximation.beastkernelmatrix` when
[`GPUMatrixData`](@ref) is passed as `matrixdata`; requires the ACABEASTCUDA package
extension to be loaded.

# Fields

  - `nearassembler::NearBlockAssemblerType`: GPU-backed BEAST assembler providing
    matrix entries

# Type parameters

  - `T`: scalar element type returned by kernel evaluations
  - `NearBlockAssemblerType`: type of the underlying GPU BEAST assembler
"""
struct GPUBEASTKernelMatrix{T,NearBlockAssemblerType} <: AbstractKernelMatrix{T}
    nearassembler::NearBlockAssemblerType
    function GPUBEASTKernelMatrix{T}(nearassembler) where {T}
        return new{T,typeof(nearassembler)}(nearassembler)
    end
end

function Base.size(matrix::GPUBEASTKernelMatrix, dim=nothing)
    if dim === nothing
        return (length(matrix.nearassembler.tfs), length(matrix.nearassembler.bfs))
    elseif dim == 1
        return length(matrix.nearassembler.tfs)
    elseif dim == 2
        return length(matrix.nearassembler.bfs)
    else
        error("dim must be either 1 or 2")
    end
end

function nextrc!(buffer, matrix::GPUBEASTKernelMatrix, rows, columns)
    return matrix(buffer, rows, columns)
end
