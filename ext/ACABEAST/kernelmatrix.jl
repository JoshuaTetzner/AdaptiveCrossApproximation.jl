function AdaptiveCrossApproximation.AbstractKernelMatrix(
    operator::BEAST.IntegralOperator,
    testspace::BEAST.Space,
    trialspace::BEAST.Space;
    matrixdata=BEAST.defaultquadstrat(operator, testspace, trialspace),
)
    return AdaptiveCrossApproximation.BEASTKernelMatrix{scalartype(operator)}(
        BEAST.blockassembler(operator, testspace, trialspace; quadstrat=matrixdata)
    )
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
