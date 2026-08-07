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

# Upper decreasing envelope of the norm history, used to clean the extrapolation input.
# Scan newest→oldest and keep a pivot only if its norm exceeds the last kept one; every
# downward dip (a redundant-pick collapse) — single- or multi-point — is dropped, so the
# fit tracks the monotone trend the residual should follow instead of being dragged down
# by transient drops. Returns the kept indices in ascending order.
function _monotone_backward_indices(v::AbstractVector, n::Int)
    idx = Int[n]
    ref = v[n]
    @inbounds for j in (n - 1):-1:1
        if v[j] > ref
            push!(idx, j)
            ref = v[j]
        end
    end
    reverse!(idx)
    return idx
end

function (convcrit::FNormExtrapolatorFunctor{F})(
    rowbuffer::AbstractMatrix{K},
    colbuffer::AbstractMatrix{K},
    npivot::Int,
    maxrows::Int,
    maxcolumns::Int,
) where {F<:Real,K}
    npivot_, conv = convcrit.estimator(rowbuffer, colbuffer, npivot, maxrows, maxcolumns)
    (npivot_ != npivot) && (return npivot_, conv)
    # Record this pivot's update norm on EVERY pivot, before branching: on the
    # non-converged branch a noisy dip must not leave a 0 in `lastnorms`, which later
    # becomes log10(0) = -Inf and corrupts the quadratic extrapolation (spurious stop).
    @views convcrit.lastnorms[npivot] =
        norm(rowbuffer[npivot, 1:maxcolumns]) * norm(colbuffer[1:maxrows, npivot])
    conv && return npivot, true
    npivot - 1 < 4 && return npivot, false
    idx = _monotone_backward_indices(convcrit.lastnorms, npivot - 1)
    length(idx) < 4 && return npivot, true
    f2 = fit(F.(idx), log10.(convcrit.lastnorms[idx]), 2)
    return npivot, f2(F(npivot)) > log10(convcrit.estimator.tol * convcrit.estimator.normUV)
end

function (convcrit::FNormExtrapolatorFunctor{F})(
    rcbuffer::AbstractVector{K}, npivot::Int
) where {F<:Real,K}
    npivot_, conv = convcrit.estimator(rcbuffer, npivot)
    (npivot_ != npivot) && (return npivot_, conv)
    # Record this pivot's update norm on EVERY pivot, before branching: on the
    # non-converged branch a noisy dip must not leave a 0 in `lastnorms`, which later
    # becomes log10(0) = -Inf and corrupts the quadratic extrapolation (spurious stop).
    length(convcrit.lastnorms) < npivot && resize!(convcrit.lastnorms, npivot)
    @inbounds convcrit.lastnorms[npivot] = norm(rcbuffer)
    conv && return npivot, true
    npivot - 1 < 4 && return npivot, false
    idx = _monotone_backward_indices(convcrit.lastnorms, npivot - 1)
    length(idx) < 4 && return npivot, true
    f2 = fit(F.(idx), log10.(convcrit.lastnorms[idx]), 2)
    return npivot, f2(F(npivot)) > log10(tolerance(convcrit) * convcrit.estimator.normUV)
end
