function AdaptiveCrossApproximation.beastkernelmatrix(
    operator::BEAST.IntegralOperator,
    testspace::BEAST.Space,
    trialspace::BEAST.Space,
    data::AdaptiveCrossApproximation.GPUMatrixData,
)
    data.ndevices == 1 ||
        throw(ArgumentError("GPUBEASTKernelMatrix currently supports one GPU device"))
    bext = Base.get_extension(BEAST, :BEASTCUDAExt)
    assembler = bext.gpu_blockassembler(
        operator,
        testspace,
        trialspace;
        quadstrat=data.quadstrat,
        device=data.device,
    )
    return AdaptiveCrossApproximation.GPUBEASTKernelMatrix{BEAST.scalartype(operator)}(
        assembler
    )
end

function (matrix::AdaptiveCrossApproximation.GPUBEASTKernelMatrix)(
    matrixblock::CUDA.AnyCuMatrix, testdata, trialdata
)
    matrix.nearassembler(matrixblock, testdata, trialdata)
    return nothing
end

function (matrix::AdaptiveCrossApproximation.GPUBEASTKernelMatrix)(
    matrixblock::AbstractMatrix, testdata, trialdata
)
    matrixblock_d = CUDA.zeros(eltype(matrix), size(matrixblock))
    matrix.nearassembler(matrixblock_d, testdata, trialdata)
    copyto!(matrixblock, matrixblock_d)
    return nothing
end
