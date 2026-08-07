"""
    BEASTKernelMatrix{T,NearBlockAssemblerType} <: AbstractKernelMatrix{T}

Kernel matrix wrapper for BEAST operator assembly.

Provides lazy matrix entry evaluation through a BEAST near-field block assembler,
which computes matrix entries on demand from operator and basis function data.

# Fields

  - `nearassembler::NearBlockAssemblerType`: BEAST assembler providing matrix entries

# Type parameters

  - `T`: scalar element type returned by kernel evaluations
  - `NearBlockAssemblerType`: type of the underlying BEAST assembler
"""
struct BEASTKernelMatrix{T,NearBlockAssemblerType} <: AbstractKernelMatrix{T}
    nearassembler::NearBlockAssemblerType
    function BEASTKernelMatrix{T}(nearassembler) where {T}
        return new{T,typeof(nearassembler)}(nearassembler)
    end
end

function Base.size(M::BEASTKernelMatrix, dim=nothing)
    return _kernelmatrix_size(length(M.nearassembler.tfs), length(M.nearassembler.bfs), dim)
end

nextrc!(buf, A::BEASTKernelMatrix, i, j) = A(buf, i, j)
