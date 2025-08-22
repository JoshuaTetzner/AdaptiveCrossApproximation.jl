struct RandomSamplingPivoting{F,K} <: ConvPivStrat
    convcrit::RandomSampling{F,K}
    rc::Int
end

function (pivstrat::RandomSamplingPivoting{F,K})(::AbstractArray) where {F<:Real,K}
    return pivstrat.convcrit.indices[argmax(pivstrat.convcrit.rest), pivstrat.rc]
end

function (pivstrat::RandomSamplingPivoting{F,K})(convergence::ConvCrit) where {F<:Real,K}
    return RandomSamplingPivoting(convergence, pivstrat.rc)
end
