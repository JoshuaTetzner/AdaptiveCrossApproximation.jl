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

# Least-squares degree-2 fit of the points (i, y[i]), i = 1..n (x = 1:n), evaluated
# at n+1. Since the abscissae are always 1..n, the fit is a 3x3 symmetric normal-
# equations solve done in scalars - same result as `Polynomials.fit(1:n, y, 2)`
# but without the per-call Vandermonde/abscissa/Polynomial allocations.
@inline function _quad_extrapolate_end(y, n::Int, ::Type{F}) where {F}
    s1 = s2 = s3 = s4 = zero(F)
    t0 = t1 = t2 = zero(F)
    @inbounds for i in 1:n
        fi = F(i)
        fi2 = fi * fi
        s1 += fi
        s2 += fi2
        s3 += fi2 * fi
        s4 += fi2 * fi2
        yi = y[i]
        t0 += yi
        t1 += fi * yi
        t2 += fi2 * yi
    end
    s0 = F(n)
    # adjugate of the symmetric normal matrix [s0 s1 s2; s1 s2 s3; s2 s3 s4]
    m11 = s2 * s4 - s3 * s3
    m12 = s3 * s2 - s1 * s4
    m13 = s1 * s3 - s2 * s2
    m22 = s0 * s4 - s2 * s2
    m23 = s2 * s1 - s0 * s3
    m33 = s0 * s2 - s1 * s1
    det = s0 * m11 + s1 * m12 + s2 * m13
    c0 = (m11 * t0 + m12 * t1 + m13 * t2) / det
    c1 = (m12 * t0 + m22 * t1 + m23 * t2) / det
    c2 = (m13 * t0 + m23 * t1 + m33 * t2) / det
    x = F(n + 1)
    return c0 + c1 * x + c2 * x * x
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
    extrap = _quad_extrapolate_end(convcrit.fitbuffer, nfit, F)
    return extrap <= log10(tolerance(convcrit) * convcrit.estimator.normUV)
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
    extrap = _quad_extrapolate_end(convcrit.fitbuffer, j, F)
    return extrap <= log10(tolerance(convcrit) * convcrit.estimator.normUV)
end

function (convcrit::PhaseExtrapolatorFunctor{F})(
    rcbuffer::AbstractVector{K}, npivot::Int
) where {F<:Real,K}
    rcnorm = norm(rcbuffer)
    rcnorm <= _zeroresidualtol(convcrit.estimator) && return npivot - 1, false
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
