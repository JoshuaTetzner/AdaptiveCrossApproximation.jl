function AdaptiveCrossApproximation.AbstractKernelMatrix(
    operator::BEAST.IntegralOperator,
    testspace::BEAST.Space,
    trialspace::BEAST.Space;
    matrixdata=BEAST.defaultquadstrat(operator, testspace, trialspace),
)
    return beastkernelmatrix(operator, testspace, trialspace, matrixdata)
end

function beastkernelmatrix(operator, testspace, trialspace, quadstrat)
    assembler = BEAST.blockassembler(operator, testspace, trialspace; quadstrat)

    return AdaptiveCrossApproximation.BEASTKernelMatrix{scalartype(operator)}(assembler)
end

function beastkernelmatrix(
    operator, testspace, trialspace, data::AdaptiveCrossApproximation.GPUMatrixData
)
    assembler = BEAST.blockassembler(operator, testspace, trialspace; data.quadstrat)

    return AdaptiveCrossApproximation.GPUBEASTKernelMatrix{scalartype(operator)}(assembler)
end

struct BlockStoreFunctor{M}
    matrix::M
end

function (f::BlockStoreFunctor)(v, m, n)
    @views f.matrix[m, n] += v
    return nothing
end

function (blk::AdaptiveCrossApproximation.BEASTKernelMatrix)(matrixblock, tdata, sdata)
    blk.nearassembler(tdata, sdata, BlockStoreFunctor(matrixblock))
    return nothing
end
