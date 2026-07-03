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
