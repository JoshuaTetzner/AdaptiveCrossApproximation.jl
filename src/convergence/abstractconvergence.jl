"""
    ConvCrit

Abstract base type for convergence criteria used by ACA and IACA compressors.

# Notes

Concrete subtypes define how stopping decisions are made and are converted into
stateful `ConvCritFunctor` objects during block compression.

# See also

`ConvCritFunctor`, `FNormEstimator`, `FNormExtrapolator`, `PhaseExtrapolator`, `RandomSampling`
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
    return throw(ArgumentError("reset! is not implemented for $(typeof(convcrit))."))
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
    @views rnorm = norm(rowbuffer[npivot, 1:maxcolumns])
    @views cnorm = norm(colbuffer[1:maxrows, npivot])
    delta_sq = (rnorm * cnorm)^2
    for j in 1:(npivot - 1)
        @views delta_sq +=
            2 * real.(
                dot(colbuffer[1:maxrows, npivot], colbuffer[1:maxrows, j]) *
                dot(rowbuffer[npivot, 1:maxcolumns], rowbuffer[j, 1:maxcolumns]),
            )
    end
    convcrit.normUV = sqrt(convcrit.normUV^2 + delta_sq)
    return nothing
end
