"""
    RandomSamplingPivoting <: ConvPivStrat

Pivoting strategy that uses the error of random sampling from the convergence estimation.

Instead of selecting pivots based on maximum values or geometric properties, this
strategy chooses pivots from randomly sampled indices used by a random sampling
convergence criterion. Works in conjunction with the random-sampling convergence functor
to provide statistically based pivot selection.

# Fields

  - `rc::Int`: Index indicating which coordinate (row=1 or column=2) to select from
"""
struct RandomSamplingPivoting <: ConvPivStrat
    rc::Int
end

struct RandomSamplingPivotingFunctor{F,K,M} <: ConvPivStratFunctor
    convcrit::RandomSamplingFunctor{F,K,M}
    rc::Int
end

function (pivstrat::RandomSamplingPivoting)(
    convcrit::RandomSamplingFunctor{F,K,M}
) where {F<:Real,K,M}
    return RandomSamplingPivotingFunctor(convcrit, pivstrat.rc)
end

function (piv::RandomSamplingPivoting)(convcrit::CombinedConvCritFunctor)
    rscrit = findfirst(x -> x isa RandomSamplingFunctor, convcrit.crits)
    if rscrit === nothing
        throw(ArgumentError("No RandomSamplingFunctor found in CombinedConvCritFunctor"))
    end
    return RandomSamplingPivotingFunctor(convcrit.crits[rscrit], piv.rc)
end

_buildpivstrat(strat::RandomSamplingPivoting, convcrit, idcs) = strat(convcrit)

function Base.resize!(pivstrat::RandomSamplingPivotingFunctor, args...)
    # Linked to RandomSamplingFunctor state; nothing to resize locally.
    return nothing
end

function reset!(pivstrat::RandomSamplingPivotingFunctor, args...)
    # Linked to RandomSamplingFunctor state; nothing to reset locally.
    return nothing
end

function (pivstrat::RandomSamplingPivotingFunctor{F,K,M})(
    ::AbstractArray
) where {F<:Real,K,M}
    nactive = pivstrat.convcrit.nactive
    rest = view(pivstrat.convcrit.rest, 1:nactive)
    indices = view(pivstrat.convcrit.indices, 1:nactive)
    return indices[argmax(abs.(rest))][pivstrat.rc]
end
