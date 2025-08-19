mutable struct FNormEstimator{F} <: ConvCrit
    normUV²::F
    tol::F
end

function FNormEstimator(tolerance::F) where {F}
    return FNormEstimator(F(0.0), tolerance)
end

function (cc::FNormEstimator{F})() where {F}
    return FNormEstimator(0.0, cc.tol)
end

tolerance(cc::FNormEstimator) = cc.tol

function (convcrit::FNormEstimator{F})(
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
