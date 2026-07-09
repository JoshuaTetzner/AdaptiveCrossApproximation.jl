"""
    MaximumValue <: ValuePivStrat

Pivoting strategy that selects the index with maximum absolute value.

This is the standard pivoting strategy used in classical ACA algorithms also referred to
as partial pivoting. At each iteration, it chooses the row or column with the largest
absolute value among the unused indices, ensuring numerical stability and good
approximation quality.
"""
struct MaximumValue <: ValuePivStrat end

mutable struct MaximumValueFunctor <: ValuePivStratFunctor
    nactive::Int
    usedidcs::Vector{Bool}
end

(::MaximumValue)(idcs::AbstractVector{<:Integer}) =
    MaximumValueFunctor(length(idcs), zeros(Bool, length(idcs)))
(::MaximumValue)(nidcs::Int) = MaximumValueFunctor(nidcs, zeros(Bool, nidcs))

function Base.resize!(pivstrat::MaximumValueFunctor, nactive::Int)
    length(pivstrat.usedidcs) < nactive && resize!(pivstrat.usedidcs, nactive)
    pivstrat.nactive = nactive
    return nothing
end

function reset!(pivstrat::MaximumValueFunctor, idcs::AbstractVector{<:Integer})
    resize!(pivstrat, length(idcs))
    @inbounds for i in 1:(pivstrat.nactive)
        pivstrat.usedidcs[i] = false
    end
    return nothing
end

function (pivstrat::MaximumValueFunctor)()
    @assert pivstrat.nactive >= 1
    pivstrat.usedidcs[1] = true
    return 1
end

function (pivstrat::MaximumValueFunctor)(rc::AbstractArray)
    nactive = pivstrat.nactive

    nextidx = 0
    maxval = 0.0
    @inbounds for i in 1:nactive
        if (!pivstrat.usedidcs[i]) && abs(rc[i]) >= maxval
            nextidx = i
            maxval = abs(rc[i])
        end
    end

    if nextidx == 0
        nextidx = 1
        maxval = abs(rc[1])
        @inbounds for i in 2:nactive
            if abs(rc[i]) >= maxval
                nextidx = i
                maxval = abs(rc[i])
            end
        end
        maxval != 0.0 && return nextidx
    end

    pivstrat.usedidcs[nextidx] = true
    return nextidx
end
