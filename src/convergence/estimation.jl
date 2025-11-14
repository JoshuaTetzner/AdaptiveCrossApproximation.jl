mutable struct FNormEstimator{F} <: ConvCrit
    tol::F
end

mutable struct FNormEstimatorFunctor{F} <: ConvCritFunctor
    normUV²::F
    tol::F
end

function (cc::FNormEstimator{F})() where {F}
    return FNormEstimatorFunctor(F(0.0), cc.tol)
end

tolerance(cc::FNormEstimator) = cc.tol

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

# iACA
mutable struct iFNormEstimator{F} <: ConvCrit
    tol::F
end

mutable struct iFNormEstimatorFunctor{F} <: ConvCritFunctor
    normUV::F
    tol::F
end

function (cc::iFNormEstimator{F})() where {F}
    return iFNormEstimatorFunctor(F(0.0), cc.tol)
end

tolerance(cc::iFNormEstimatorFunctor) = cc.tol

function (convcrit::iFNormEstimatorFunctor{F})(
    rcbuffer::AbstractVector{K}, npivot::Int
) where {F<:Real,K}
    @views rcnorm = norm(rcbuffer)

    isapprox(rcnorm, 0.0) && (return npivot - 1, false)
    return npivot, rcnorm > tolerance(convcrit) * convcrit.normUV
end
