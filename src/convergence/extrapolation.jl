mutable struct FNormExtrapolator{F} <: ConvCrit
    lastnorms::Vector{F}
    estimator::FNormEstimator{F}
end

function FNormExtrapolator(tol::F) where {F}
    return FNormExtrapolator(F[], FNormEstimator(F(0.0), tol))
end

function (cc::FNormExtrapolator{F})() where {F}
    return FNormExtrapolator(F[], FNormEstimator(0.0, cc.estimator.tol))
end
tolerance(cc::FNormExtrapolator) = cc.estimator.tol

# ACA
function (convcrit::FNormExtrapolator{F})(
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
