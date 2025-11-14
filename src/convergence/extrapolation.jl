using Polynomials

mutable struct FNormExtrapolator{F} <: ConvCrit
    estimator::Union{FNormEstimator{F},iFNormEstimator{F}}
end

mutable struct FNormExtrapolatorFunctor{F} <: ConvCritFunctor
    lastnorms::Vector{F}
    estimator::Union{FNormEstimatorFunctor{F},iFNormEstimatorFunctor{F}}
end

function FNormExtrapolator(tol::F) where {F}
    return FNormExtrapolator(FNormEstimator(tol))
end

function (cc::FNormExtrapolator{F})() where {F}
    return FNormExtrapolatorFunctor(F[], cc.estimator())
end
tolerance(cc::FNormExtrapolatorFunctor) = cc.estimator.tol

# ACA
function (convcrit::FNormExtrapolatorFunctor{F})(
    rowbuffer::AbstractMatrix{K},
    colbuffer::AbstractMatrix{K},
    npivot::Int,
    maxrows::Int,
    maxcolumns::Int,
) where {F<:Real,K}
    npivot_, conv = convcrit.estimator(rowbuffer, colbuffer, npivot, maxrows, maxcolumns)
    (npivot_ != npivot) && (return npivot_, conv)
    (!conv) && (f2 = fit(Vector(1:(npivot - 1)), log10.(convcrit.lastnorms), 2))
    @views push!(
        convcrit.lastnorms,
        norm(rowbuffer[npivot, 1:maxcolumns]) * norm(colbuffer[1:maxrows, npivot]),
    )
    if conv
        return npivot, true
    else
        return npivot,
        f2(npivot) > log10(convcrit.estimator.tol * sqrt(convcrit.estimator.normUV²))
    end
end

# iACA
function (convcrit::FNormExtrapolatorFunctor{F})(
    rcbuffer::AbstractVector{K}, npivot::Int
) where {F<:Real,K}
    npivot_, conv = convcrit.estimator(rcbuffer, npivot)
    (npivot_ != npivot) && (return npivot_, conv)

    (!conv) && (f2 = fit(Vector(1:(npivot - 1)), log10.(convcrit.lastnorms), 2))
    @views push!(convcrit.lastnorms, norm(rcbuffer))
    if conv
        return npivot, true
    else
        return npivot, f2(npivot) > log10(tolerance(convcrit) * convcrit.estimator.normUV)
    end
end
