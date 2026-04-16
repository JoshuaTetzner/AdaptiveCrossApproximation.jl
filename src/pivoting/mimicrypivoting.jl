"""
    MimicryPivoting{D,F<:Real} <: GeoPivStrat

Geometric pivoting strategy that mimics point distribution of a fully pivoted ACA geometrically.

Selects pivots to reproduce the spatial distribution of a fully pivoted ACA.
The strategy balances three objectives: geometric separation (Leja-like behavior),
proximity to the reference distribution, and fill distance maximization. Particularly
useful for H²--matrix compression where incomplete factorizations are sufficient.

# Fields

  - `refpos::Vector{SVector{D,F}}`: Positions of test or expansion domain
  - `pos::Vector{SVector{D,F}}`: Positions from which to select pivots

# Type Parameters

  - `D`: Spatial dimension
  - `F`: Floating point type for coordinates
"""
struct MimicryPivoting{D,F<:Real} <: GeoPivStrat
    refpos::Vector{SVector{D,F}}
    pos::Vector{SVector{D,F}}
end

@inline function bestindex(
    leja::AbstractVector{F},
    h::AbstractVector{F},
    w::AbstractVector{F},
    nactive::Int,
    npivot::Int,
) where {F<:Real}
    nactive > 0 || throw(ArgumentError("nactive must be positive."))
    npivot > 1 || throw(ArgumentError("npivot must be larger than 1."))

    exponent = F(2) / F(npivot - 1)
    @inbounds begin
        nextlocal = 1
        bestscore = (leja[1]^exponent) * h[1] * (w[1]^4)
        for i in 2:nactive
            score = (leja[i]^exponent) * h[i] * (w[i]^4)
            if score > bestscore
                bestscore = score
                nextlocal = i
            end
        end
        return nextlocal
    end
end

mutable struct MimicryPivotingFunctor{D,F<:Real} <: GeoPivStratFunctor
    pivoting::MimicryPivoting{D,F}
    nactive::Int
    refcentroid::SVector{D,F}
    idcs::Vector{Int}
    h::Vector{F}
    leja::Vector{F}
    w::Vector{F}
end

function (strat::MimicryPivoting{D,F})(refidcs, idcs) where {D,F}
    nactive = length(idcs)
    refcentroid = _centroid(strat.refpos, refidcs)
    idcs = collect(idcs)
    h = zeros(F, nactive)
    w = zeros(F, nactive)
    leja = ones(F, nactive)

    @inbounds for i in 1:nactive
        w[i] = 1 / norm(strat.pos[idcs[i]] - refcentroid)
    end

    return MimicryPivotingFunctor{D,F}(strat, nactive, refcentroid, idcs, h, leja, w)
end

_buildpivstrat(strat::MimicryPivoting, refidcs, idcs, maxrank) = strat(refidcs, idcs)

function Base.resize!(functor::MimicryPivotingFunctor{D,F}, nactive::Int) where {D,F<:Real}
    if length(functor.idcs) < nactive
        resize!(functor.idcs, nactive)
        resize!(functor.h, nactive)
        resize!(functor.leja, nactive)
        resize!(functor.w, nactive)
    end
    functor.nactive = nactive
    return nothing
end

function reset!(
    functor::MimicryPivotingFunctor{D,F},
    refidcs::AbstractVector{Int},
    idcs::AbstractVector{Int},
) where {D,F<:Real}
    resize!(functor, length(idcs))
    functor.refcentroid = _centroid(functor.pivoting.refpos, refidcs)
    pos = functor.pivoting.pos
    @inbounds for i in 1:(functor.nactive)
        functor.idcs[i] = idcs[i]
        functor.h[i] = zero(F)
        functor.leja[i] = one(F)
        functor.w[i] = 1 / norm(pos[functor.idcs[i]] - functor.refcentroid)
    end
    return nothing
end

function (pivstrat::MimicryPivotingFunctor{D,F})() where {D,F}
    AdaptiveCrossApproximation.leja2_init!(pivstrat, pivstrat.idcs[1], pivstrat.nactive)
    return 1
end

function (pivstrat::MimicryPivotingFunctor{D,F})(rc::AbstractArray) where {D,F}
    nactive = pivstrat.nactive
    if all(iszero, view(pivstrat.h, 1:nactive))
        AdaptiveCrossApproximation.leja2_init!(pivstrat, pivstrat.idcs[1], nactive)
    end

    nextidx = bestindex(
        view(pivstrat.leja, 1:nactive),
        view(pivstrat.h, 1:nactive),
        view(pivstrat.w, 1:nactive),
        nactive,
        1,
    )

    AdaptiveCrossApproximation.leja2!(pivstrat, pivstrat.idcs[nextidx], nactive)
    return nextidx
end
