"""
    _select_free_devices(n) -> Vector{Int}

Pick `n` currently-idle CUDA device ids (no compute processes from *other* processes,
checked via NVML), preferring lower ids. Contexts held by the current process are
ignored, so repeated builds in one session reuse their own devices instead of marching
across the node. Throws if fewer than `n` devices are free - shared-node GPU usage
shifts over time, so this is only a check at build time, not a reservation.
"""
function _select_free_devices(n::Int)
    self = getpid()
    free = Int[
        i for i in 0:(CUDA.ndevices() - 1) if
        all(==(self), keys(CUDA.NVML.compute_processes(CUDA.NVML.Device(i))))
    ]
    length(free) >= n || throw(ArgumentError(
        "requested $n free GPU device(s), only $(length(free)) of $(CUDA.ndevices()) " *
        "are currently idle (busy: $(setdiff(0:(CUDA.ndevices() - 1), free)))"
    ))
    return free[1:n]
end

function AdaptiveCrossApproximation.beastkernelmatrix(
    operator::BEAST.IntegralOperator,
    testspace::BEAST.Space,
    trialspace::BEAST.Space,
    data::AdaptiveCrossApproximation.GPUMatrixData,
)
    devices = data.devices !== nothing ? data.devices : _select_free_devices(data.ndevices)
    bext = Base.get_extension(BEAST, :BEASTCUDAExt)
    functors = bext.gpu_blockassemblers(
        operator, testspace, trialspace; quadstrat=data.quadstrat, devices=devices
    )
    return AdaptiveCrossApproximation.GPUBEASTKernelMatrix{BEAST.scalartype(operator, testspace, trialspace)}(
        functors
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
