using Polynomials

"""
    FNormExtrapolator{F} <: ConvCrit

Convergence criterion using polynomial extrapolation of pivot norms.
Combines norm estimation with quadratic extrapolation to predict convergence.

# Fields

  - `estimator::Union{FNormEstimator{F},iFNormEstimator{F}}`: Underlying norm estimator
"""
mutable struct FNormExtrapolator{F} <: ConvCrit
    estimator::Union{FNormEstimator{F},iFNormEstimator{F}}
end

"""
    FNormExtrapolatorFunctor{F} <: ConvCritFunctor

Stateful extrapolator tracking pivot norm history.
Fits quadratic polynomial to log-scaled norms for convergence prediction.

# Fields

  - `lastnorms::Vector{F}`: History of pivot norms for extrapolation
  - `estimator::Union{FNormEstimatorFunctor{F},iFNormEstimatorFunctor{F}}`: Active estimator functor
"""
mutable struct FNormExtrapolatorFunctor{F} <: ConvCritFunctor
    lastnorms::Vector{F}
    estimator::Union{FNormEstimatorFunctor{F},iFNormEstimatorFunctor{F}}
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

"""
    (cc::FNormExtrapolator{F})()

Initialize extrapolator functor with empty history.
"""
function (cc::FNormExtrapolator{F})() where {F}
    return FNormExtrapolatorFunctor(F[], cc.estimator())
end

"""
    tolerance(cc::FNormExtrapolatorFunctor)

Get tolerance from underlying estimator.
"""
tolerance(cc::FNormExtrapolatorFunctor) = cc.estimator.tol

"""
    (convcrit::FNormExtrapolatorFunctor)(rowbuffer, colbuffer, npivot, maxrows, maxcolumns)

Check convergence for ACA using extrapolation.
Fits quadratic to log-norms and extrapolates to predict convergence.

# Arguments

  - `rowbuffer::AbstractMatrix{K}`: Row factor buffer
  - `colbuffer::AbstractMatrix{K}`: Column factor buffer
  - `npivot::Int`: Current pivot index
  - `maxrows::Int`: Number of active rows
  - `maxcolumns::Int`: Number of active columns

# Returns

  - `npivot::Int`: Final pivot count
  - `continue::Bool`: Whether to continue iteration
"""
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

"""
    (convcrit::FNormExtrapolatorFunctor)(rcbuffer::AbstractVector{K}, npivot::Int)

Check convergence for iACA using extrapolation.
Applies extrapolation to incomplete ACA norm history.

# Arguments

  - `rcbuffer::AbstractVector{K}`: Current row or column buffer
  - `npivot::Int`: Current pivot index

# Returns

  - `npivot::Int`: Final pivot count
  - `continue::Bool`: Whether to continue iteration
"""
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
