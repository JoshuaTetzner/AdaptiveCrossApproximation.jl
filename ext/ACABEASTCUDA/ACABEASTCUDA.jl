module ACABEASTCUDA

using AdaptiveCrossApproximation
using BEAST
using BlockSparseMatrices
using CUDA

# BEAST's GPU block assembly lives in its BEASTCUDAExt, which released BEAST
# versions do not provide yet. Look it up at call time and fail clearly.
function _beastcudaext()
    ext = Base.get_extension(BEAST, :BEASTCUDAExt)
    isnothing(ext) && throw(
        ArgumentError(
            "GPUMatrixData requires a BEAST version with GPU block assembly " *
            "(BEASTCUDAExt), which is not available in the loaded BEAST.",
        ),
    )
    return ext
end

include("kernelmatrix.jl")
include("nearinteractions.jl")

end
