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
