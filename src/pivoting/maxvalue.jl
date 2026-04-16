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
    fill!(view(pivstrat.usedidcs, 1:(pivstrat.nactive)), false)
    return nothing
end

function (pivstrat::MaximumValueFunctor)()
    @assert pivstrat.nactive >= 1
    pivstrat.usedidcs[1] = true
    return 1
end

function (pivstrat::MaximumValueFunctor)(rc::AbstractArray)
    nactive = pivstrat.nactive
    used = view(pivstrat.usedidcs, 1:nactive)

    if all(used)
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
