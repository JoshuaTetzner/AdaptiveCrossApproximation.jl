module ACABEAST

using BEAST
using AdaptiveCrossApproximation

include("kernelmatrix.jl")

function AdaptiveCrossApproximation.defaultfarmatrixdata(
    operator::BEAST.IntegralOperator, testspace::BEAST.Space, trialspace::BEAST.Space
)
    return BEAST.DoubleNumQStrat(2, 3)
end

function AdaptiveCrossApproximation.defaultmatrixdata(
    operator::BEAST.IntegralOperator, testspace::BEAST.Space, trialspace::BEAST.Space
)
    return BEAST.defaultquadstrat(operator, testspace, trialspace)
end

function AdaptiveCrossApproximation.defaultcompressor(
    ::BEAST.IntegralOperator, ::BEAST.Space, ::BEAST.Space; tol::Float64=1e-4
)
    return AdaptiveCrossApproximation.ACA(; tol=tol)
end

function AdaptiveCrossApproximation.defaultcompressor(
    ::Union{
        BEAST.MWDoubleLayer3D,
        BEAST.HH3DDoubleLayerFDBIO,
        BEAST.HH3DDoubleLayerTransposedFDBIO,
        BEAST.HH2DDoubleLayerFDBIO,
    },
    ::BEAST.Space,
    ::BEAST.Space;
    tol::Float64=1e-4,
)
    c1 = AdaptiveCrossApproximation.FNormEstimator(tol)
    c2 = AdaptiveCrossApproximation.RandomSampling(; tol=tol)
    convergence = AdaptiveCrossApproximation.CombinedConvCrit([c1, c2])
    rowpivoting = AdaptiveCrossApproximation.CombinedPivStrat([
        MaximumValue(), AdaptiveCrossApproximation.RandomSamplingPivoting(1)
    ])

    compressor = ACA(; rowpivoting=rowpivoting, convergence=convergence)
    return compressor
end

function AdaptiveCrossApproximation.scalartype(operator::BEAST.IntegralOperator)
    return BEAST.scalartype(operator)
end

function Base.permute!(space::BEAST.Space, permutation::Vector{Int})
    permute!(space.fns, permutation)
    permute!(space.pos, permutation)
    return nothing
end
end
