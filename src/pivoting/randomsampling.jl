struct RandomSamplingPivoting <: ConvPivStrat
    #convcrit::RandomSampling{F}
    rc::Int
end

struct RandomSamplingPivotingFunctor{F,K} <: ConvPivStratFunctor
    convcrit::RandomSamplingFunctor{F,K}
    rc::Int
end

function (pivstrat::RandomSamplingPivotingFunctor{F,K})(::AbstractArray) where {F<:Real,K}
    return pivstrat.convcrit.indices[argmax(pivstrat.convcrit.rest), pivstrat.rc]
end

function (pivstrat::RandomSamplingPivoting)(
    convcrit::RandomSamplingFunctor{F,K}
) where {F<:Real,K}
    return RandomSamplingPivotingFunctor(convcrit, pivstrat.rc)
end
