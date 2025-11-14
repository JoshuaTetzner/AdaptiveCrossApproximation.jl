
struct MaximumValue <: ValuePivStrat end

struct MaximumValueFunctor <: ValuePivStratFunctor
    usedidcs::Vector{Bool}
end

(::MaximumValue)(len::Int) = MaximumValueFunctor(zeros(Bool, len))
(::MaximumValue)(idcs::AbstractArray{Int}) = MaximumValueFunctor(zeros(Bool, length(idcs)))

function (pivstrat::MaximumValueFunctor)()
    pivstrat.usedidcs[1] = true
    return 1
end

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
