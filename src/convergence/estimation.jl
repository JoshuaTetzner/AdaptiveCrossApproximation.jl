mutable struct FNormEstimator{F} <: ConvCrit
    normUV²::F
end

function FNormEstimator(::Type{F}) where {F}
    return FNormEstimator(F(0.0))
end

(::FNormEstimator{F})() where {F} = FNormEstimator(0.0)

function (convcrit::FNormEstimator{F})(
    rowbuffer::AbstractMatrix{K},
    colbuffer::AbstractMatrix{K},
    npivot::Int,
    maxrows::Int,
    maxcolumns::Int,
    tol::F,
) where {F<:Real,K}
    @views rnorm = norm(rowbuffer[npivot, 1:maxcolumns])
    @views cnorm = norm(colbuffer[1:maxrows, npivot])

    (isapprox(rnorm, 0.0) && isapprox(cnorm, 0.0)) && (return npivot - 1, false)
    if (isapprox(rnorm, 0.0) || isapprox(cnorm, 0.0))
        (npivot == 1) ? (return npivot - 1, true) : (return npivot - 1, false)
    end
    normF!(convcrit, rowbuffer, colbuffer, npivot, maxrows, maxcolumns)
    return npivot, rnorm * cnorm > tol * sqrt(convcrit.normUV²)
end

# iACA
mutable struct iFNormEstimator{F} <: ConvCrit
    normUV::F
end

function iFNormEstimator(::Type{F}) where {F}
    return iFNormEstimator(F(0.0))
end

function (convcrit::iFNormEstimator{F})(
    rcbuffer::AbstractVector{K}, npivot::Int, tol::F
) where {F<:Real,K}
    @views rcnorm = norm(rcbuffer)

    isapprox(rcnorm, 0.0) && (return npivot - 1, true)

    return npivot, rcnorm > tol * convcrit.normUV
end
