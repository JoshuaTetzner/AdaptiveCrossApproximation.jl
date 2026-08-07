"""
    PhaseExtrapolator{F} <: ConvCrit

Convergence criterion using polynomial extrapolation of pivot norms across multiple phases.

Extends [`FNormExtrapolator`](@ref) with per-phase direction tracking: convergence requires
that all active phases individually show extrapolated convergence, and that two consecutive
pivots both satisfy the criterion.

# Fields

  - `estimator::FNormEstimator{F}`: Underlying norm estimator

# See also

`FNormExtrapolator`, `PhaseExtrapolatorFunctor`, `reset_phase!`
"""
struct PhaseExtrapolator{F} <: ConvCrit
    estimator::FNormEstimator{F}
end

"""
    PhaseExtrapolator(tol::F)

Construct a phase extrapolator with the given tolerance.
"""
function PhaseExtrapolator(tol::F) where {F<:Real}
    return PhaseExtrapolator(FNormEstimator(tol))
end

"""
    PhaseExtrapolatorFunctor{F} <: ConvCritFunctor

Stateful functor for [`PhaseExtrapolator`](@ref).

Tracks per-pivot norms and their associated phases, fitting a quadratic polynomial
to predict convergence. Requires two consecutive converging pivots before stopping.

# Fields

  - `estimator::FNormEstimatorFunctor{F}`: Moving-average norm state
  - `lastnorms::Vector{F}`: Per-pivot norms, indexed by pivot number
  - `normdirections::Vector{Int32}`: Phase label for each pivot
  - `lastconverged::Bool`: Whether the previous pivot also converged
  - `currentdirection::Int32`: Phase label currently active (set via `reset_phase!`)
"""
mutable struct PhaseExtrapolatorFunctor{F} <: ConvCritFunctor
    estimator::FNormEstimatorFunctor{F}
    lastnorms::Vector{F}
    normdirections::Vector{Int32}
    fitbuffer::Vector{F}
    lastconverged::Bool
    currentdirection::Int32
end

function (cc::PhaseExtrapolator{F})() where {F<:Real}
    return cc(0)
end

function (cc::PhaseExtrapolator{F})(maxrank::Int) where {F<:Real}
    return PhaseExtrapolatorFunctor(
        cc.estimator(),
        zeros(F, maxrank),
        zeros(Int32, maxrank),
        Vector{F}(),
        false,
        typemin(Int32),
    )
end

_buildconvcrit(cc::PhaseExtrapolator, A, rowidcs, colidcs, maxrank) = cc(maxrank)

function reset!(convcrit::PhaseExtrapolatorFunctor)
    reset!(convcrit.estimator)
    fill!(convcrit.lastnorms, zero(eltype(convcrit.lastnorms)))
    fill!(convcrit.normdirections, Int32(0))
    convcrit.lastconverged = false
    convcrit.currentdirection = typemin(Int32)
    return nothing
end

tolerance(cc::PhaseExtrapolator) = cc.estimator.tol
tolerance(cc::PhaseExtrapolatorFunctor) = cc.estimator.tol

"""
    reset_phase!(convcrit::PhaseExtrapolatorFunctor, direction::Integer)

Set the current phase label. Call before each new pivot to associate it with a direction.
"""
function reset_phase!(convcrit::PhaseExtrapolatorFunctor, direction::Integer)
    convcrit.currentdirection = Int32(direction)
    return nothing
end

function _extrapolated_converged(
    convcrit::PhaseExtrapolatorFunctor{F}, npivot::Int
) where {F}
    nfit = npivot - 1
    nfit < 3 && return true
    length(convcrit.fitbuffer) < nfit && resize!(convcrit.fitbuffer, nfit)
    @inbounds for i in 1:nfit
        convcrit.fitbuffer[i] = log10(convcrit.lastnorms[i])
    end
    f2 = fit(F.(1:nfit), view(convcrit.fitbuffer, 1:nfit), 2)
    return f2(F(nfit + 1)) <= log10(tolerance(convcrit) * convcrit.estimator.normUV)
end

function _direction_extrapolated_converged(
    convcrit::PhaseExtrapolatorFunctor{F}, npivot::Int
) where {F}
    dir = convcrit.currentdirection
    dir <= 0 && return true
    nmax = npivot - 1
    length(convcrit.fitbuffer) < nmax && resize!(convcrit.fitbuffer, nmax)
    j = 0
    @inbounds for i in 1:nmax
        convcrit.normdirections[i] == dir || continue
        j += 1
        convcrit.fitbuffer[j] = log10(convcrit.lastnorms[i])
    end
    j < 3 && return true
    f2 = fit(F.(1:j), view(convcrit.fitbuffer, 1:j), 2)
    return f2(F(j + 1)) <= log10(tolerance(convcrit) * convcrit.estimator.normUV)
end

function (convcrit::PhaseExtrapolatorFunctor{F})(
    rcbuffer::AbstractVector{K}, npivot::Int
) where {F<:Real,K}
    rcnorm = norm(rcbuffer)
    isapprox(rcnorm, 0.0) && return npivot - 1, false
    length(convcrit.lastnorms) < npivot && resize!(convcrit.lastnorms, npivot)
    length(convcrit.normdirections) < npivot && resize!(convcrit.normdirections, npivot)
    convcrit.lastnorms[npivot] = rcnorm
    convcrit.normdirections[npivot] = convcrit.currentdirection
    converged =
        rcnorm <= tolerance(convcrit) * convcrit.estimator.normUV &&
        _extrapolated_converged(convcrit, npivot) &&
        _direction_extrapolated_converged(convcrit, npivot)
    keepgoing = !(converged && convcrit.lastconverged)
    convcrit.lastconverged = converged
    return npivot, keepgoing
end
