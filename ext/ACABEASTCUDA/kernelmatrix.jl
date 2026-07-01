function AdaptiveCrossApproximation.beastkernelmatrix(
    operator::BEAST.IntegralOperator,
    testspace::BEAST.Space,
    trialspace::BEAST.Space,
    data::AdaptiveCrossApproximation.GPUMatrixData,
)
    assembler = BEAST.blockassembler(operator, testspace, trialspace; data.quadstrat)
    return AdaptiveCrossApproximation.GPUBEASTKernelMatrix{BEAST.scalartype(operator)}(
        assembler
    )
end

struct BlockStoreFunctor{M}
    matrix::M
end

function (store::BlockStoreFunctor)(value, row, column)
    @views store.matrix[row, column] += value
    return nothing
end

function (matrix::AdaptiveCrossApproximation.GPUBEASTKernelMatrix)(
    matrixblock, testdata, trialdata
)
    matrix.nearassembler(testdata, trialdata, BlockStoreFunctor(matrixblock))
    return nothing
end
