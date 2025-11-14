
"""
    MaximumValue <: ValuePivStrat

Pivoting strategy that selects the index with maximum absolute value.

This is the standard pivoting strategy used in classical ACA algorithms also referred to
as partial pivoting. At each iteration, it chooses the row or column with the largest
absolute value among the unused indices, ensuring numerical stability and good
approximation quality.
"""
struct MaximumValue <: ValuePivStrat end

"""
    MaximumValueFunctor <: ValuePivStratFunctor

Stateful functor that tracks which indices have been used during pivot selection.

Created by calling a [`MaximumValue`](@ref) instance with length or index information.
Maintains a boolean vector to ensure each index is selected at most once.

# Fields

  - `usedidcs::Vector{Bool}`: Tracks which indices have been selected as pivots
"""
struct MaximumValueFunctor <: ValuePivStratFunctor
    usedidcs::Vector{Bool}
end

"""
    (::MaximumValue)(idcs::AbstractArray{Int})

Create a `MaximumValueFunctor` for the given index array.

Returns a functor with tracking vector sized to match the length of `idcs`.
"""
(::MaximumValue)(idcs::AbstractArray{Int}) = MaximumValueFunctor(zeros(Bool, length(idcs)))

"""
    (pivstrat::MaximumValueFunctor)()

Select the first index as the initial pivot.

Returns `1` and marks it as used. Used when no row/column data is available yet.
"""
function (pivstrat::MaximumValueFunctor)()
    pivstrat.usedidcs[1] = true
    return 1
end

"""
    (pivstrat::MaximumValueFunctor)(rc::AbstractArray)

Select the unused index with maximum absolute value in `rc`.

Searches through all unused indices, finds the one with largest `abs(rc[i])`,
marks it as used, and returns its index.

# Arguments

  - `rc::AbstractArray`: Row or column data to select from

# Returns

  - `nextidx::Int`: Index of the maximum absolute value among unused indices
"""
function (pivstrat::MaximumValueFunctor)(rc::AbstractArray)
    nextidx = 1
    maxval = 0.0
    for i in eachindex(pivstrat.usedidcs)
        if (!pivstrat.usedidcs[i]) && abs(rc[i]) >= maxval
            nextidx = i
            maxval = abs(rc[i])
        end
    end

    pivstrat.usedidcs[nextidx] = true
    return nextidx
end
