
struct CombinedPivStrat
    convcrit::CombinedConvCrit
    strats::Vector{PivStrat}
end

function (pivstrat::CombinedPivStrat)()
    return pivstrat.strats[1]()
end

function (pivstrat::CombinedPivStrat)(rc::AbstractArray)
    #öprintln(pivstrat.convcrit.isconverged)
    length(pivstrat.strats) > length(pivstrat.convcrit.isconverged) &&
        push!(pivstrat.convcrit.isconverged, false)
    for (i, conv) in enumerate(pivstrat.convcrit.isconverged)
        !conv && continue
        i > length(pivstrat.convcrit.crits) && pivstrat.convcrit.isconverged[i] == true
        return pivstrat.strats[i](rc)
    end
end
