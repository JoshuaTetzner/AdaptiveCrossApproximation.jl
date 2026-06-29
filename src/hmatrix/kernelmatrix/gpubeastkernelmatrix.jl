
struct GPUBEASTKernelMatrix{T,NearBlockAssemblerType} <: AbstractKernelMatrix{T}
    nearassembler::NearBlockAssemblerType
    function GPUBEASTKernelMatrix{T}(nearassembler) where {T}
        return new{T,typeof(nearassembler)}(nearassembler)
    end
end

struct GPUMatrixData{QuadStratType}
    quadstrat::QuadStratType
    ndevices::Int
    function GPUMatrixData(quadstrat, ndevices::Int)
        return new{typeof(quadstrat)}(quadstrat, ndevices)
    end
end

function Base.size(M::GPUBEASTKernelMatrix, dim=nothing)
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

nextrc!(buf, A::GPUBEASTKernelMatrix, i, j) = A(buf, i, j)
