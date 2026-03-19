
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
mutable struct MaximumValueFunctor <: ValuePivStratFunctor
    nactive::Int
    usedidcs::Vector{Bool}
end

"""
    (::MaximumValue)(idcs::AbstractVector{<:Integer})

Create a `MaximumValueFunctor` for the given index array.

Returns a functor with tracking vector sized to match the length of `idcs`.
"""
(::MaximumValue)(idcs::AbstractVector{<:Integer}) =
    MaximumValueFunctor(length(idcs), zeros(Bool, length(idcs)))

function Base.resize!(pivstrat::MaximumValueFunctor, nactive::Int)
    length(pivstrat.usedidcs) < nactive && resize!(pivstrat.usedidcs, nactive)
    pivstrat.nactive = nactive
    return nothing
end

function reset!(pivstrat::MaximumValueFunctor, idcs::AbstractVector{<:Integer})
    resize!(pivstrat, length(idcs))
    fill!(view(pivstrat.usedidcs, 1:(pivstrat.nactive)), false)
    return nothing
end

"""
    (pivstrat::MaximumValueFunctor)()

Select the first index as the initial pivot.

Returns `1` and marks it as used. Used when no row/column data is available yet.
"""
function (pivstrat::MaximumValueFunctor)()
    @assert pivstrat.nactive >= 1
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
    nactive = pivstrat.nactive
    used = view(pivstrat.usedidcs, 1:nactive)

    if all(used)
        @warn "Rectangular full-rank blockstructure detected."
        absrx = abs.(view(rc, 1:nactive))
        maximum(absrx) != 0.0 && (return argmax(absrx))
    end

    nextidx = 1
    maxval = 0.0
    for i in 1:nactive
        if (!pivstrat.usedidcs[i]) && abs(rc[i]) >= maxval
            nextidx = i
            maxval = abs(rc[i])
        end
    end

    pivstrat.usedidcs[nextidx] = true
    return nextidx
end
