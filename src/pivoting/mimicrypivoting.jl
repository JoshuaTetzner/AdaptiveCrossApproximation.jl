
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

  - `pos::Vector{SVector{D,F}}`: All geometric positions
  - `idcs::Vector{Int}`: Current indices being considered for selection
  - `h::Vector{F}`: Minimum distances from each point to selected points (fill distance)
  - `leja::Vector{F}`: Product of distances to all selected points (Leja metric)
  - `w::Vector{F}`: Weights based on inverse distance to reference centroid
"""
mutable struct MimicryPivotingFunctor{D,F<:Real} <: GeoPivStratFunctor
    pos::Vector{SVector{D,F}}
    idcs::Vector{Int}
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
    ref = sum(strat.refpos[refidcs]) ./ length(refidcs)
    h = zeros(F, length(rcidcs))
    w = 1 ./ norm.(strat.pos[rcidcs] .- Scalar(ref))
    leja = ones(F, length(rcidcs))

    return MimicryPivotingFunctor{D,F}(strat.pos, rcidcs, h, leja, w)
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
    nextidx = strat.idcs[argmax(strat.w)]
    @views strat.h .= norm.(strat.pos[strat.idcs] .- Scalar(strat.pos[nextidx]))
    strat.leja .*= norm.(strat.pos[strat.idcs] .- Scalar(strat.pos[nextidx]))
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
    nextidx = strat.idcs[argmax(
        strat.leja .^ (2 / (npivot - 1)) .* strat.h .* strat.w .^ 4
    )]
    leja2!(strat, nextidx)
    strat.leja .*= norm.(strat.pos[strat.idcs] .- Scalar(strat.pos[nextidx]))
    return nextidx
end
