"""
    FNormEstimator{F} <: ConvCrit

Frobenius norm-based convergence criterion for standard ACA.
Stops iteration when relative error estimate falls below tolerance.

# Fields

  - `tol::F`: Relative tolerance threshold
"""
mutable struct FNormEstimator{F} <: ConvCrit
    tol::F
end

"""
    FNormEstimatorFunctor{F} <: ConvCritFunctor

Stateful Frobenius norm estimator for ACA compression.
Tracks squared norm of UV factorization across iterations.

# Fields

  - `normUV²::F`: Accumulated squared Frobenius norm of UV
  - `tol::F`: Relative tolerance threshold
"""
mutable struct FNormEstimatorFunctor{F} <: ConvCritFunctor
    normUV²::F
    tol::F
end

"""
    (cc::FNormEstimator{F})()

Initialize FNormEstimator functor with zero accumulated norm.
"""
function (cc::FNormEstimator{F})() where {F}
    return FNormEstimatorFunctor(F(0.0), cc.tol)
end

"""
    tolerance(cc::FNormEstimator)

Get tolerance threshold from estimator.
"""
tolerance(cc::FNormEstimator) = cc.tol

"""
    (convcrit::FNormEstimatorFunctor)(rowbuffer, colbuffer, npivot, maxrows, maxcolumns)

Check convergence for standard ACA using Frobenius norm estimate.
Returns (npivot, continue) where continue is true if iteration should proceed.

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
function (convcrit::FNormEstimatorFunctor{F})(
    rowbuffer::AbstractMatrix{K},
    colbuffer::AbstractMatrix{K},
    npivot::Int,
    maxrows::Int,
    maxcolumns::Int,
) where {F<:Real,K}
    @views rnorm = norm(rowbuffer[npivot, 1:maxcolumns])
    @views cnorm = norm(colbuffer[1:maxrows, npivot])

    (isapprox(rnorm, 0.0) && isapprox(cnorm, 0.0)) && (return npivot - 1, false)
    if (isapprox(rnorm, 0.0) || isapprox(cnorm, 0.0))
        (npivot == 1) ? (return npivot - 1, true) : (return npivot - 1, false)
    end
    normF!(convcrit, rowbuffer, colbuffer, npivot, maxrows, maxcolumns)
    return npivot, rnorm * cnorm > convcrit.tol * sqrt(convcrit.normUV²)
end

"""
    iFNormEstimator{F} <: ConvCrit

Frobenius norm-based convergence criterion for incomplete ACA (iACA).
Uses moving average norm estimate for geometric pivoting scenarios.

# Fields

  - `tol::F`: Relative tolerance threshold
"""
mutable struct iFNormEstimator{F} <: ConvCrit
    tol::F
end

"""
    iFNormEstimatorFunctor{F} <: ConvCritFunctor

Stateful Frobenius norm estimator for iACA compression.
Tracks moving average of row/column norms.

# Fields

  - `normUV::F`: Moving average norm
  - `tol::F`: Relative tolerance threshold
"""
mutable struct iFNormEstimatorFunctor{F} <: ConvCritFunctor
    normUV::F
    tol::F
end

"""
    (cc::iFNormEstimator{F})()

Initialize iFNormEstimator functor with zero accumulated norm.
"""
function (cc::iFNormEstimator{F})() where {F}
    return iFNormEstimatorFunctor(F(0.0), cc.tol)
end

"""
    tolerance(cc::iFNormEstimatorFunctor)

Get tolerance threshold from iACA estimator functor.
"""
tolerance(cc::iFNormEstimatorFunctor) = cc.tol

"""
    (convcrit::iFNormEstimatorFunctor)(rcbuffer::AbstractVector{K}, npivot::Int)

Check convergence for iACA using moving average norm.
Returns (npivot, continue) where continue is true if iteration should proceed.

# Arguments

  - `rcbuffer::AbstractVector{K}`: Current row or column buffer
  - `npivot::Int`: Current pivot index

# Returns

  - `npivot::Int`: Final pivot count
  - `continue::Bool`: Whether to continue iteration
"""
function (convcrit::iFNormEstimatorFunctor{F})(
    rcbuffer::AbstractVector{K}, npivot::Int
) where {F<:Real,K}
    @views rcnorm = norm(rcbuffer)

    isapprox(rcnorm, 0.0) && (return npivot - 1, false)
    return npivot, rcnorm > tolerance(convcrit) * convcrit.normUV
end
