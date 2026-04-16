"""
    CombinedPivStrat

Composite pivoting strategy that switches between multiple strategies based on convergence.

Combines multiple pivoting strategies with a combined convergence criterion, allowing
the pivot selection method to change as different convergence criteria are satisfied.
For example, can start with geometric pivoting and switch to value-based pivoting
once a certain accuracy is reached.

# Fields

  - `strats::Vector{PivStrat}`: Ordered list of pivoting strategies to use
"""
struct CombinedPivStrat <: PivStrat
    strats::Vector{PivStrat}
end

struct CombinedPivStratFunctor <: PivStratFunctor
    convcrit::CombinedConvCritFunctor
    strats::Vector{PivStratFunctor}
end

function (pivstrat::CombinedPivStrat)(
    convergence::CombinedConvCritFunctor, idcs::AbstractArray{Int}
)
    curr_strats = Vector{PivStratFunctor}(undef, length(pivstrat.strats))
    for (i, strat) in enumerate(pivstrat.strats)
        if isa(strat, RandomSamplingPivoting)
            curr_strats[i] = strat(convergence)
        else
            curr_strats[i] = strat(idcs)
        end
    end

    return CombinedPivStratFunctor(convergence, curr_strats)
end

_buildpivstrat(strat::CombinedPivStrat, convcrit, idcs) = strat(convcrit, idcs)

function Base.resize!(pivstrat::CombinedPivStratFunctor, args...)
    for strat in pivstrat.strats
        resize!(strat, args...)
    end
    return nothing
end

function reset!(pivstrat::CombinedPivStratFunctor, args...)
    for strat in pivstrat.strats
        reset!(strat, args...)
    end
    return nothing
end

function (pivstrat::CombinedPivStratFunctor)()
    return pivstrat.strats[1]()
end

function (pivstrat::CombinedPivStratFunctor)(rc::AbstractArray)
    length(pivstrat.strats) > length(pivstrat.convcrit.isconverged) &&
        push!(pivstrat.convcrit.isconverged, false)
    for (i, conv) in enumerate(pivstrat.convcrit.isconverged)
        !conv && continue
        i > length(pivstrat.convcrit.crits) && pivstrat.convcrit.isconverged[i] == true
        nextidx = pivstrat.strats[i](rc)

        if !(pivstrat.strats[i] isa MaximumValueFunctor)
            mvidx = findfirst(x -> x isa MaximumValueFunctor, pivstrat.strats)
            mvidx !== nothing && (pivstrat.strats[mvidx].usedidcs[nextidx] = true)
        end

        return nextidx
    end
end
