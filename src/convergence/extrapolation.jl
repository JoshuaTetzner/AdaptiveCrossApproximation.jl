mutable struct FNormExtrapolator{F} <: ConvCrit
    lastnorms::Vector{F}
    estimator::Union{FNormEstimator{F},iFNormEstimator{F}}
end

function FNormExtrapolator(::Type{F}) where {F}
    return FNormExtrapolator(F[], FNormEstimator(F(0.0)))
end

(::FNormExtrapolator{F})() where {F} = FNormExtrapolator(F[], FNormEstimator(0.0))

# ACA
function (convcrit::FNormExtrapolator{F})(
    rowbuffer::AbstractMatrix{K},
    colbuffer::AbstractMatrix{K},
    npivot::Int,
    maxrows::Int,
    maxcolumns::Int,
    tol::F,
) where {F<:Real,K}
    npivot_, conv = convcrit.estimator(
        rowbuffer, colbuffer, npivot, maxrows, maxcolumns, tol
    )
    (npivot_ != npivot) && (return npivot_, conv)
    (!conv) && (f2 = fit(Vector(1:(npivot - 1)), log10.(convcrit.lastnorms), 2))
    @views push!(
        convcrit.lastnorms,
        norm(rowbuffer[npivot, 1:maxcolumns]) * norm(colbuffer[1:maxrows, npivot]),
    )
    if conv
        return npivot, true
    else
        return npivot, f2(npivot) > log10(tol * sqrt(convcrit.estimator.normUV²))
    end
end

# iACA
function (convcrit::FNormExtrapolator{F})(
    rcbuffer::AbstractVector{K}, npivot::Int, tol::F
) where {F<:Real,K}
    npivot_, conv = convcrit.estimator(rcbuffer, npivot, tol)
    (npivot_ != npivot) && (return npivot_, conv)

    (!conv) && (f2 = fit(Vector(1:(npivot - 1)), log10.(convcrit.lastnorms), 2))
    @views push!(convcrit.lastnorms, norm(rcbuffer))
    if conv
        return npivot, true
    else
        return npivot, f2(npivot) > log10(tol * convcrit.estimator.normUV)
    end
end
