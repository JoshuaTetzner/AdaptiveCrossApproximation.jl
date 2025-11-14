"""
    RandomSamplingPivoting <: ConvPivStrat

Pivoting strategy that uses the error of random sampling from the convergence estimation.

Instead of selecting pivots based on maximum values or geometric properties, this
strategy chooses pivots from randomly sampled indices used by a random sampling
convergence criterion. Works in conjunction with [`RandomSamplingFunctor`](@ref)
to provide statistically based pivot selection.

# Fields

  - `rc::Int`: Index indicating which coordinate (row=1 or column=2) to select from
"""
struct RandomSamplingPivoting <: ConvPivStrat
    rc::Int
end

"""
    RandomSamplingPivotingFunctor{F,K} <: ConvPivStratFunctor

Stateful functor for random sampling-based pivot selection.

Links to a random sampling convergence criterion functor to access the randomly
sampled indices and their residuals, selecting pivots from the worst-performing
samples.

# Fields

  - `convcrit::RandomSamplingFunctor{F,K}`: Convergence criterion with sample information
  - `rc::Int`: Coordinate index (1 for row, 2 for column)
"""
struct RandomSamplingPivotingFunctor{F,K} <: ConvPivStratFunctor
    convcrit::RandomSamplingFunctor{F,K}
    rc::Int
end

function (piv::RandomSamplingPivoting)(convcrit::CombinedConvCritFunctor)
    rscrit = findfirst(x -> x isa RandomSamplingFunctor, convcrit.crits)
    if rscrit === nothing
        throw(ArgumentError("No RandomSamplingFunctor found in CombinedConvCritFunctor"))
    end
    return RandomSamplingPivotingFunctor(convcrit.crits[rscrit], piv.rc)
end

"""
    (pivstrat::RandomSamplingPivotingFunctor{F,K})(::AbstractArray)

Select pivot from the sample with largest residual.

Examines the random samples tracked by the convergence criterion and returns the
row or column index (depending on `rc`) corresponding to the sample with the
maximum residual error.

# Arguments

  - `::AbstractArray`: Row/column data (unused, selection based on random samples)

# Returns

  - Index from the worst-performing random sample
"""
function (pivstrat::RandomSamplingPivotingFunctor{F,K})(::AbstractArray) where {F<:Real,K}
    return pivstrat.convcrit.indices[argmax(pivstrat.convcrit.rest), pivstrat.rc]
end

"""
    (pivstrat::RandomSamplingPivoting)(convcrit::RandomSamplingFunctor{F,K})

Create a `RandomSamplingPivotingFunctor` linked to the convergence criterion.

Initializes the functor by connecting it to the random sampling convergence
criterion that tracks sampled indices and residuals.

# Arguments

  - `convcrit::RandomSamplingFunctor{F,K}`: Random sampling convergence functor

# Returns

  - `RandomSamplingPivotingFunctor`: Initialized functor linked to the criterion
"""
function (pivstrat::RandomSamplingPivoting)(
    convcrit::RandomSamplingFunctor{F,K}
) where {F<:Real,K}
    return RandomSamplingPivotingFunctor(convcrit, pivstrat.rc)
end
