"""
    GPUBEASTKernelMatrix{T,NearBlockAssemblerType} <: AbstractKernelMatrix{T}

Kernel matrix backed by a GPU-resident BEAST near-field assembler.

Built by the `ACABEASTCUDA` extension from a BEAST operator and space pair; entries
are evaluated by delegating to `nearassembler`, which handles device transfer and
batched GPU quadrature internally.
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
