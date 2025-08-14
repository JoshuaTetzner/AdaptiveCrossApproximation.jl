
struct CombinedPivStrat
    convcrit::CombinedConvCrit
    strats::Vector{PivStrat}
end

function (pivstrat::CombinedPivStrat)()
    return pivstrat.strats[1]()
end

function (pivstrat::CombinedPivStrat)(rc::AbstractArray)
    #println(pivstrat.convcrit.isconverged)
    length(pivstrat.strats) > length(pivstrat.convcrit.isconverged) &&
        push!(pivstrat.convcrit.isconverged, false)
    for (i, conv) in enumerate(pivstrat.convcrit.isconverged)
        !conv && continue
        i > length(pivstrat.convcrit.crits) && pivstrat.convcrit.isconverged[i] == true
        return pivstrat.strats[i](rc)
    end
end

function (pivstrat::CombinedPivStrat)(convergence::CombinedConvCrit, idcs::Vector{Int})
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
