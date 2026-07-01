struct GPUBEASTKernelMatrix{T,NearBlockAssemblerType} <:
       AdaptiveCrossApproximation.AbstractKernelMatrix{T}
    nearassembler::NearBlockAssemblerType
    function GPUBEASTKernelMatrix{T}(nearassembler) where {T}
        return new{T,typeof(nearassembler)}(nearassembler)
    end
end

function AdaptiveCrossApproximation.beastkernelmatrix(
    operator::BEAST.IntegralOperator,
    testspace::BEAST.Space,
    trialspace::BEAST.Space,
    data::AdaptiveCrossApproximation.GPUMatrixData,
)
    assembler = BEAST.blockassembler(operator, testspace, trialspace; data.quadstrat)
    return GPUBEASTKernelMatrix{BEAST.scalartype(operator)}(assembler)
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

struct BlockStoreFunctor{M}
    matrix::M
end

function (store::BlockStoreFunctor)(value, row, column)
    @views store.matrix[row, column] += value
    return nothing
end

function (matrix::GPUBEASTKernelMatrix)(matrixblock, testdata, trialdata)
    matrix.nearassembler(testdata, trialdata, BlockStoreFunctor(matrixblock))
    return nothing
end

function AdaptiveCrossApproximation.nextrc!(
    buffer, matrix::GPUBEASTKernelMatrix, rows, columns
)
    return matrix(buffer, rows, columns)
end
