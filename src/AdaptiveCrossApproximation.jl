module AdaptiveCrossApproximation

using LinearAlgebra
using StaticArrays

include("utils.jl")

include("hmatrix/kernelmatrix/abstractkernelmatrix.jl")
include("hmatrix/kernelmatrix/beastkernelmatrix.jl")
include("hmatrix/kernelmatrix/pointmatrix.jl")

include("pivoting/abstractpivoting.jl")
include("convergence/abstractconvergence.jl")

include("pivoting/maxvalue.jl")
include("pivoting/lejapoints.jl")
include("pivoting/filldistance.jl")
include("pivoting/mimicrypivoting.jl")
include("pivoting/treemimicrypivoting.jl")

include("convergence/estimation.jl")
include("convergence/extrapolation.jl")
include("convergence/randomsampling.jl")
include("convergence/combinedconvcrit.jl")

include("pivoting/combinedpivstrat.jl")
include("pivoting/randomsampling.jl")

nextrc!(buf, A::AbstractArray, i, j) = (buf .= view(A, i, j))

include("aca.jl")
#include("acaT.jl")
include("iaca.jl")

if !isdefined(Base, :get_extension) # for julia version < 1.9
    include("../ext/ACAH2Trees/ACAH2Trees.jl")
end

include("hmatrix/abstracthmatrix.jl")

module H
    using ..AdaptiveCrossApproximation: HMatrix, _tree, H2Tree

    function assemble(op, space; args...)
        return error("Not implemented")
    end

    function assemble(
        op,
        testspace,
        trialspace;
        tree=_tree(
            H2Tree(), testspace, trialspace, 1 / 2^10; minvaluestest=200, minvaluestrial=200
        ),
        kwargs...,
    )
        return HMatrix(op, testspace, trialspace, tree; kwargs...)
    end
end

export H
export ACA
export iACA
export FNormEstimator, iFNormEstimator, FNormExtrapolator
export MaximumValue, Leja2, FillDistance
export MimicryPivoting, TreeMimicryPivoting
export reset!
export AbstractKernelMatrix
end
