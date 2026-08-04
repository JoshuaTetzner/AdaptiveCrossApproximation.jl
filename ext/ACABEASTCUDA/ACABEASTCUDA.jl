module ACABEASTCUDA

using AdaptiveCrossApproximation
using BEAST
using BlockSparseMatrices
using CUDA

include("kernelmatrix.jl")
include("nearinteractions.jl")

end
