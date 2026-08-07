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

# Build each sub-strategy through its own `_buildpivstrat`, exactly as a standalone
# pivoting would be built: value strategies consume `idcs` (a candidate-index vector in
# the NCA path, or a candidate count in the standalone ACA/HMatrix path — `MaximumValue`
# handles both), while convergence-driven ones such as `RandomSamplingPivoting` consume
# the shared `convergence`. This keeps `CombinedPivStrat` a generic strategy combiner,
# like `CombinedConvCrit`, rather than assuming a fixed set of strategies or an index
# vector.
function (pivstrat::CombinedPivStrat)(convergence::CombinedConvCritFunctor, idcs)
    curr_strats = Vector{PivStratFunctor}(undef, length(pivstrat.strats))
    for (i, strat) in enumerate(pivstrat.strats)
        curr_strats[i] = _buildpivstrat(strat, convergence, idcs)
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
        nextidx = pivstrat.strats[i](rc)

        if !(pivstrat.strats[i] isa MaximumValueFunctor)
            mvidx = findfirst(x -> x isa MaximumValueFunctor, pivstrat.strats)
            mvidx !== nothing && (pivstrat.strats[mvidx].usedidcs[nextidx] = true)
        end

        return nextidx
    end
    return throw(ArgumentError("No converged strategy found in CombinedPivStratFunctor."))
end
