struct MaximumValue <: PivStrat
    usedidcs::Vector{Bool}
end

MaximumValue() = MaximumValue(Bool[])

(::MaximumValue)(rcp::Vector{Int}) = MaximumValue(zeros(Bool, length(rcp)))

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
