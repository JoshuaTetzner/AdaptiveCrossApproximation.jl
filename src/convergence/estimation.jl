"""
    FNormEstimator{F} <: ConvCrit

Frobenius norm-based convergence criterion for ACA and IACA.

Dispatches on input type: matrix arguments follow the ACA path (accumulated Frobenius
norm), vector arguments follow the IACA path (moving-average norm).

# Fields

  - `tol::F`: Relative tolerance threshold
"""
mutable struct FNormEstimator{F} <: ConvCrit
    tol::F
end

mutable struct FNormEstimatorFunctor{F} <: ConvCritFunctor
    normUV::F
    tol::F
end

function (cc::FNormEstimator{F})() where {F}
    return FNormEstimatorFunctor(F(0.0), cc.tol)
end

_buildconvcrit(cc::FNormEstimator, A, rowidcs, colidcs, maxrank) = cc()

function reset!(convcrit::FNormEstimatorFunctor)
    convcrit.normUV = zero(convcrit.normUV)
    return nothing
end

tolerance(cc::FNormEstimator) = cc.tol
tolerance(cc::FNormEstimatorFunctor) = cc.tol

function normF!(
    convcrit::FNormEstimatorFunctor, rcbuffer::AbstractVector{K}, npivot::Int
) where {K}
    convcrit.normUV = ((npivot - 1) * convcrit.normUV + norm(rcbuffer)) / npivot
    return nothing
end

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
    return npivot, rnorm * cnorm > convcrit.tol * convcrit.normUV
end

function (convcrit::FNormEstimatorFunctor{F})(
    rcbuffer::AbstractVector{K}, npivot::Int
) where {F<:Real,K}
    rcnorm = norm(rcbuffer)
    isapprox(rcnorm, 0.0) && (return npivot - 1, false)
    return npivot, rcnorm > convcrit.tol * convcrit.normUV
end
