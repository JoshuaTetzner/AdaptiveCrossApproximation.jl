module ACABEAST

using BEAST
using BEAST.CompScienceMeshes: cells, vertices
using LinearAlgebra
using AdaptiveCrossApproximation

include("kernelmatrix.jl")
include("rwgorientations.jl")

function AdaptiveCrossApproximation.tofarquadstrat(qs::BEAST.DoubleNumWiltonSauterQStrat)
    return BEAST.DoubleNumQStrat(qs.outer_rule_far, qs.inner_rule_far)
end

function AdaptiveCrossApproximation.defaultmatrixdata(
    operator::BEAST.IntegralOperator, testspace::BEAST.Space, trialspace::BEAST.Space
)
    return BEAST.defaultquadstrat(operator, testspace, trialspace)
end

function AdaptiveCrossApproximation.defaultcompressor(
    ::BEAST.IntegralOperator, ::BEAST.Space, ::BEAST.Space; tol::Real=1e-4
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
    tol::Real=1e-4,
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

"""
    assemble(operator::BEAST.LocalOperator, testspace::BEAST.Space, trialspace::BEAST.Space; kwargs...)

Assemble a BEAST local operator (e.g. `Identity`, `NCross`) directly, bypassing
tree construction and ACA compression.

`LocalOperator`s only couple overlapping basis functions, so their Galerkin
matrix is cheap to build and never benefits from low-rank approximation.
Compression-only keywords (`tol`, `maxrank`, `tree`, ...) accepted by the
generic `assemble` are ignored here; only `quadstrat` is forwarded to
`BEAST.assemble`.
"""
function AdaptiveCrossApproximation.assemble(
    operator::BEAST.LocalOperator,
    testspace::BEAST.Space,
    trialspace::BEAST.Space;
    quadstrat=BEAST.defaultquadstrat(operator, testspace, trialspace),
    kwargs...,
)
    return BEAST.assemble(operator, testspace, trialspace; quadstrat=quadstrat)
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
