
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
mutable struct MimicryPivoting{D,F<:Real} <: GeoPivStrat
    refpos::Vector{SVector{D,F}}
    pos::Vector{SVector{D,F}}
end

"""
    MimicryPivotingFunctor{D,F<:Real} <: GeoPivStratFunctor

Functor for mimicry-based pivot selection.

Maintains vectors for leja2 (h), leja (leja), and weights (w) based on distance to reference centroid.

# Fields

    - `refpos::Vector{SVector{D,F}}`: Reference positions used to compute centroid weights

  - `pos::Vector{SVector{D,F}}`: All geometric positions
  - `idcs::Vector{Int}`: Current indices being considered for selection
  - `nactive::Int`: Active prefix length in state vectors
  - `h::Vector{F}`: Minimum distances from each point to selected points (fill distance)
  - `leja::Vector{F}`: Product of distances to all selected points (Leja metric)
  - `w::Vector{F}`: Weights based on inverse distance to reference centroid
"""
mutable struct MimicryPivotingFunctor{D,F<:Real} <: GeoPivStratFunctor
    refpos::Vector{SVector{D,F}}
    pos::Vector{SVector{D,F}}
    idcs::Vector{Int}
    nactive::Int
    h::Vector{F}
    leja::Vector{F}
    w::Vector{F}
end

"""
    (strat::MimicryPivoting{D,F})(refidcs, rcidcs)

Create a `MimicryPivotingFunctor` for the given reference and candidate indices.

Initializes the functor by computing the centroid of the reference points and
setting up weights that favor points close to this centroid. This encourages
the selected pivots to spatially mimic the reference distribution.

# Arguments

  - `refidcs`: Indices of reference points (e.g., parent cluster pivots)
  - `rcidcs`: Indices of candidate points to select from (e.g., child cluster points)

# Returns

  - `MimicryPivotingFunctor`: Initialized functor with computed weights and metrics
"""
function (strat::MimicryPivoting{D,F})(refidcs, rcidcs) where {D,F}
    nactive = length(rcidcs)
    ref = _centroid(strat.refpos, refidcs)
    idcs = collect(rcidcs)
    h = zeros(F, nactive)
    w = zeros(F, nactive)
    leja = ones(F, nactive)

    @inbounds for i in 1:nactive
        w[i] = 1 / norm(strat.pos[idcs[i]] - ref)
    end

    return MimicryPivotingFunctor{D,F}(strat.refpos, strat.pos, idcs, nactive, h, leja, w)
end

function Base.resize!(
    functor::MimicryPivotingFunctor{D,F}, nactive::Integer
) where {D,F<:Real}
    nactive < 0 && throw(ArgumentError("nactive must be non-negative"))
    resize!(functor.idcs, nactive)
    resize!(functor.h, nactive)
    resize!(functor.leja, nactive)
    resize!(functor.w, nactive)
    functor.nactive = nactive
    return functor
end

function reset!(
    functor::MimicryPivotingFunctor{D,F},
    refidcs::AbstractVector{<:Integer},
    rcidcs::AbstractVector{<:Integer},
) where {D,F<:Real}
    nactive = length(rcidcs)
    length(functor.idcs) < nactive && resize!(functor, nactive)
    functor.nactive = nactive

    ref = _centroid(functor.refpos, refidcs)
    @inbounds for i in 1:nactive
        idx = Int(rcidcs[i])
        functor.idcs[i] = idx
        functor.h[i] = zero(F)
        functor.leja[i] = one(F)
        functor.w[i] = 1 / norm(functor.pos[idx] - ref)
    end

    return functor
end

function reset!(
    functor::MimicryPivotingFunctor{D,F},
    strat::MimicryPivoting{D,F},
    refidcs::AbstractVector{<:Integer},
    rcidcs::AbstractVector{<:Integer},
) where {D,F<:Real}
    functor.refpos = strat.refpos
    functor.pos = strat.pos
    return reset!(functor, refidcs, rcidcs)
end

"""
    (strat::MimicryPivotingFunctor{D,F})()

Select the first pivot based on proximity to reference centroid.

Chooses the point with maximum weight (closest to the reference centroid),
then initializes distance metrics for subsequent pivot selection.

# Returns

  - Global index of the selected pivot point
"""
function (strat::MimicryPivotingFunctor{D,F})() where {D,F<:Real}
    nactive = strat.nactive
    nextlocal = argmax(view(strat.w, 1:nactive))
    nextidx = strat.idcs[nextlocal]

    AdaptiveCrossApproximation.leja2_init!(strat, nextidx, nactive)
    @inbounds for i in 1:nactive
        strat.leja[i] = strat.h[i]
    end

    return nextidx
end

"""
    (strat::MimicryPivotingFunctor{D,F})(npivot::Int)

Select next pivot balancing Leja separation, fill distance, and reference proximity.

Uses a composite metric that combines:

  - Leja product (geometric separation from all selected points)
  - Fill distance (maximum minimum distance criterion)
  - Reference weights (proximity to target distribution)

The balance between these factors evolves with iteration number `npivot`.

# Arguments

  - `npivot::Int`: Current pivot iteration number (influences weight balance)

# Returns

  - Global index of the selected pivot point
"""
function (strat::MimicryPivotingFunctor{D,F})(npivot::Int) where {D,F<:Real}
    nactive = strat.nactive
    exponent = F(2) / F(npivot - 1)

    nextlocal = 1
    bestscore = -Inf
    @inbounds for i in 1:nactive
        score = (strat.leja[i]^exponent) * strat.h[i] * (strat.w[i]^4)
        if score > bestscore
            bestscore = score
            nextlocal = i
        end
    end

    nextidx = strat.idcs[nextlocal]
    nextpos = strat.pos[nextidx]
    @inbounds for i in 1:nactive
        d = norm(strat.pos[strat.idcs[i]] - nextpos)
        if strat.h[i] > d
            strat.h[i] = d
        end
        strat.leja[i] *= d
    end

    return nextidx
end
