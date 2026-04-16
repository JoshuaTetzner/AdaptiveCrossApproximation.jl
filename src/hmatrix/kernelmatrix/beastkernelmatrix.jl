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
    if dim === nothing
        return (length(M.nearassembler.tfs), length(M.nearassembler.bfs))
    elseif dim == 1
        return length(M.nearassembler.tfs)
    elseif dim == 2
        return length(M.nearassembler.bfs)
    else
        error("dim must be either 1 or 2")
    end
end

nextrc!(buf, A::BEASTKernelMatrix, i, j) = A(buf, i, j)
