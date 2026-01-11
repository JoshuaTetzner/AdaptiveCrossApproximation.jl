module ACABEAST

using BEAST
using AdaptiveCrossApproximation

include("kernelmatrix.jl")

function permute!(space::BEAST.RTBasis, permutation::Vector{Int})
    permute!(space.fns, permutation)
    return permute!(space.pos, permutation)
end
end
