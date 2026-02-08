module ACABEAST

using BEAST
using AdaptiveCrossApproximation

include("kernelmatrix.jl")

function AdaptiveCrossApproximation.defaultfarquadstrat(
    operator::BEAST.IntegralOperator, testspace::BEAST.Space, trialspace::BEAST.Space
)
    return BEAST.DoubleNumQStrat(2, 3)
end

function AdaptiveCrossApproximation.defaultnearquadstrat(
    operator::BEAST.IntegralOperator, testspace::BEAST.Space, trialspace::BEAST.Space
)
    return BEAST.defaultquadstrat(operator, testspace, trialspace)
end

function AdaptiveCrossApproximation.scalartype(operator::BEAST.IntegralOperator)
    return BEAST.scalartype(operator)
end

function AdaptiveCrossApproximation.permute!(space::BEAST.Space, permutation::Vector{Int})
    permute!(space.fns, permutation)
    permute!(space.pos, permutation)
    return nothing
end
end
