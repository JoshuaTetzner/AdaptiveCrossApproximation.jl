"""
    ConvCrit

Abstract base type for convergence criteria used by ACA and iACA compressors.

# Notes
Concrete subtypes define how stopping decisions are made and are converted into
stateful `ConvCritFunctor` objects during block compression.

# See also
`ConvCritFunctor`, `FNormEstimator`, `iFNormEstimator`, `FNormExtrapolator`, `RandomSampling`
"""
abstract type ConvCrit end

"""
    ConvCritFunctor

Abstract base type for stateful convergence criterion functors.

# Notes
Instances are called during ACA iterations and return `(npivot, continue::Bool)`.
Subtypes should implement `reset!` to reinitialize internal state for a new block.

# See also
`ConvCrit`, `reset!`, `normF!`
"""
abstract type ConvCritFunctor end

function (cc::ConvCrit)(
    K::Union{AbstractMatrix,AbstractKernelMatrix},
    rowidcs::AbstractArray{Int},
    colidcs::AbstractArray{Int};
    maxrank::Int=40,
)
    if isa(cc, CombinedConvCrit)
        return cc(K, rowidcs, colidcs; maxrank=maxrank)
    elseif isa(cc, RandomSampling)
        return cc(K, rowidcs, colidcs)
    elseif isa(cc, FNormExtrapolator)
        return cc(maxrank)
    else
        return cc()
    end
end

"""
    reset!(convcrit::ConvCritFunctor)

Reset a convergence functor before starting compression of a new block.

# Notes
Concrete subtypes should overload this method. The default fallback throws
`ArgumentError`.

# See also
`ConvCritFunctor`
"""
function reset!(convcrit::ConvCritFunctor)
    throw(ArgumentError("reset! is not implemented for $(typeof(convcrit))."))
end

function reset!(convcrit::ConvCritFunctor, args...)
    return reset!(convcrit)
end

function normF!(
    convcrit::ConvCritFunctor,
    rowbuffer::AbstractMatrix{K},
    colbuffer::AbstractMatrix{K},
    npivot::Int,
    maxrows::Int,
    maxcolumns::Int,
) where {K}
    @views convcrit.normUV² +=
        (norm(rowbuffer[npivot, 1:maxcolumns]) * norm(colbuffer[1:maxrows, npivot]))^2

    for j in 1:(npivot - 1)
        @views convcrit.normUV² +=
            2 *
            real.(
                dot(colbuffer[1:maxrows, npivot], colbuffer[1:maxrows, j]) *
                dot(rowbuffer[npivot, 1:maxcolumns], rowbuffer[j, 1:maxcolumns]),
            )
    end
end

function normF!(
    convcrit::ConvCritFunctor, rcbuffer::AbstractVector{K}, npivot::Int
) where {K}
    return convcrit.normUV = ((npivot - 1) * convcrit.normUV + norm(rcbuffer)) / npivot
end
