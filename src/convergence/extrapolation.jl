"""
    FNormExtrapolator{F} <: ConvCrit

Convergence criterion using polynomial extrapolation of pivot norms.
Combines norm estimation with quadratic extrapolation to predict convergence.

# Fields

  - `estimator::FNormEstimator{F}`: Underlying norm estimator
"""
struct FNormExtrapolator{F} <: ConvCrit
    estimator::FNormEstimator{F}
end

struct FNormExtrapolatorFunctor{F} <: ConvCritFunctor
    lastnorms::Vector{F}
    estimator::FNormEstimatorFunctor{F}
end

"""
    FNormExtrapolator(tol::F)

Construct extrapolator with Frobenius norm estimator.

# Arguments

  - `tol::F`: Convergence tolerance
"""
function FNormExtrapolator(tol::F) where {F}
    return FNormExtrapolator(FNormEstimator(tol))
end

function (cc::FNormExtrapolator{F})(maxrank::Int) where {F<:Real}
    return FNormExtrapolatorFunctor(zeros(F, maxrank), cc.estimator())
end

_buildconvcrit(cc::FNormExtrapolator, A, rowidcs, colidcs, maxrank) = cc(maxrank)

function reset!(convcrit::FNormExtrapolatorFunctor)
    fill!(convcrit.lastnorms, zero(eltype(convcrit.lastnorms)))
    reset!(convcrit.estimator)
    return nothing
end

tolerance(cc::FNormExtrapolatorFunctor) = cc.estimator.tol
tolerance(cc::FNormExtrapolator) = cc.estimator.tol

function (convcrit::FNormExtrapolatorFunctor{F})(
    rowbuffer::AbstractMatrix{K},
    colbuffer::AbstractMatrix{K},
    npivot::Int,
    maxrows::Int,
    maxcolumns::Int,
) where {F<:Real,K}
    npivot_, conv = convcrit.estimator(rowbuffer, colbuffer, npivot, maxrows, maxcolumns)
    (npivot_ != npivot) && (return npivot_, conv)
    if conv
        @views convcrit.lastnorms[npivot] =
            norm(rowbuffer[npivot, 1:maxcolumns]) * norm(colbuffer[1:maxrows, npivot])
        return npivot, true
    else
        npivot - 1 < 3 && return npivot, false
        f2 = fit(Vector(1:(npivot - 1)), log10.(convcrit.lastnorms[1:(npivot - 1)]), 2)
        return npivot,
        f2(npivot) > log10(convcrit.estimator.tol * convcrit.estimator.normUV)
    end
end

function (convcrit::FNormExtrapolatorFunctor{F})(
    rcbuffer::AbstractVector{K}, npivot::Int
) where {F<:Real,K}
    npivot_, conv = convcrit.estimator(rcbuffer, npivot)
    (npivot_ != npivot) && (return npivot_, conv)
    if conv
        length(convcrit.lastnorms) < npivot && resize!(convcrit.lastnorms, npivot)
        @inbounds convcrit.lastnorms[npivot] = norm(rcbuffer)
        return npivot, true
    else
        npivot - 1 < 3 && return npivot, false
        f2 = fit(Vector(1:(npivot - 1)), log10.(convcrit.lastnorms[1:(npivot - 1)]), 2)
        return npivot,
        f2(F(npivot)) > log10(tolerance(convcrit) * convcrit.estimator.normUV)
    end
end
