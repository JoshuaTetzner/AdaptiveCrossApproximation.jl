
struct MaximumValue <: ValuePivStrat
    usedidcs::Vector{Bool}
end

MaximumValue() = MaximumValue(Bool[])

(::MaximumValue)(len::Int) = MaximumValue(zeros(Bool, len))
(::MaximumValue)(ivec::Vector{Int}) = MaximumValue(zeros(Bool, length(ivec)))

function (pivstrat::MaximumValue)()
    pivstrat.usedidcs[1] = true
    return 1
end

function (pivstrat::MaximumValue)(rc::AbstractArray)
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
