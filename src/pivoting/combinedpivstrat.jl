
"""
    CombinedPivStrat

Composite pivoting strategy that switches between multiple strategies based on convergence.

Combines multiple pivoting strategies with a combined convergence criterion, allowing
the pivot selection method to change as different convergence criteria are satisfied.
For example, can start with geometric pivoting and switch to value-based pivoting
once a certain accuracy is reached.

# Fields

  - `convcrit::CombinedConvCrit`: Combined convergence criterion tracking which sub-criteria are met
  - `strats::Vector{PivStrat}`: Ordered list of pivoting strategies to use
"""
struct CombinedPivStrat
    convcrit::CombinedConvCrit
    strats::Vector{PivStrat}
end

"""
    (pivstrat::CombinedPivStrat)()

Select initial pivot using the first strategy.

Delegates to the first strategy in the list for initial pivot selection when no
data is available yet.

# Returns

  - Initial pivot index from the first strategy
"""
function (pivstrat::CombinedPivStrat)()
    return pivstrat.strats[1]()
end

"""
    (pivstrat::CombinedPivStrat)(rc::AbstractArray)

Select next pivot using the first strategy whose convergence criterion is met.

Iterates through the convergence criteria and uses the strategy corresponding to
the first satisfied criterion. If no criteria are met, uses the first strategy.
Automatically updates the convergence tracking state.

# Arguments

  - `rc::AbstractArray`: Row or column data for pivot selection

# Returns

  - Pivot index selected by the active strategy
"""
function (pivstrat::CombinedPivStrat)(rc::AbstractArray)
    length(pivstrat.strats) > length(pivstrat.convcrit.isconverged) &&
        push!(pivstrat.convcrit.isconverged, false)
    for (i, conv) in enumerate(pivstrat.convcrit.isconverged)
        !conv && continue
        i > length(pivstrat.convcrit.crits) && pivstrat.convcrit.isconverged[i] == true
        return pivstrat.strats[i](rc)
    end
end

"""
    (pivstrat::CombinedPivStrat)(convergence::CombinedConvCrit, idcs::AbstractArray{Int})

Create a combined pivoting functor for the given index subset.

Initializes all constituent strategies with the provided indices and links them to
the combined convergence criterion. Handles special cases like `RandomSamplingPivoting`
which requires the convergence criterion functor rather than indices.

# Arguments

  - `convergence::CombinedConvCrit`: Combined convergence criterion
  - `idcs::AbstractArray{Int}`: Indices for the submatrix

# Returns

  - `CombinedPivStrat`: Initialized combined strategy with functors for all sub-strategies
"""
function (pivstrat::CombinedPivStrat)(
    convergence::CombinedConvCrit, idcs::AbstractArray{Int}
)
    curr_strats = Vector{PivStrat}(undef, length(pivstrat.strats))
    for (i, strat) in enumerate(pivstrat.strats)
        if isa(strat, RandomSamplingPivoting)
            curr_strats[i] = strat(convergence.crits[i])
        else
            curr_strats[i] = strat(idcs)
        end
    end
    return CombinedPivStrat(convergence, curr_strats)
end
