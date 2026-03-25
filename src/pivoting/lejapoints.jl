
"""
    Leja2{D,F<:Real} <: GeoPivStrat

Geometric pivoting strategy based on Leja points (product of distances).

A modified more efficient version of the fill distance approach.
This leads to well-separated point sequences.
These points have been introduced as modified leja points and will, therefore,
be referred to as Leja2 points within this package.

# Fields

  - `pos::Vector{SVector{D,F}}`: Geometric positions of all points (D-dimensional)

# Type Parameters

  - `D`: Spatial dimension
  - `F`: Floating point type for coordinates
"""
struct Leja2{D,F<:Real} <: GeoPivStrat
    pos::Vector{SVector{D,F}}
end

"""
    Leja2Functor{D,F<:Real} <: PivStratFunctor

Stateful functor for modified leja point pivot selection.

Maintains minimum distances from each point to all selected points, which are
updated incrementally as new pivots are chosen.

# Fields

    - `pivoting::Leja2{D,F}`: Immutable strategy carrying geometric positions
    - `nactive::Int`: Active prefix length in state vectors
    - `idcs::Vector{Int}`: Indices of points being considered for selection
    - `h::Vector{F}`: Current minimum distance from each point to selected points
"""
mutable struct Leja2Functor{D,F<:Real} <: GeoPivStratFunctor
    pivoting::Leja2{D,F}
    nactive::Int
    idcs::Vector{Int}
    h::Vector{F}
end

function (pivstrat::Leja2{D,F})(idcs::AbstractVector{<:Integer}) where {D,F}
    nactive = length(idcs)
    return Leja2Functor{D,F}(pivstrat, nactive, collect(Int, idcs), zeros(F, nactive))
end

function (pivstrat::Leja2{D,F})(nidcs::Int) where {D,F}
    return Leja2Functor{D,F}(pivstrat, nidcs, zeros(Int, nidcs), zeros(F, nidcs))
end

@inline _positions(pivstrat::GeoPivStratFunctor) = pivstrat.pivoting.pos

function Base.resize!(pivstrat::Leja2Functor{D,F}, nactive::Int) where {D,F<:Real}
    length(pivstrat.h) < nactive && resize!(pivstrat.h, nactive)
    length(pivstrat.idcs) < nactive && resize!(pivstrat.idcs, nactive)
    pivstrat.nactive = nactive
    return nothing
end

function reset!(
    pivstrat::Leja2Functor{D,F}, idcs::AbstractVector{<:Integer}
) where {D,F<:Real}
    nactive = length(idcs)
    resize!(pivstrat, nactive)

    @inbounds for i in 1:nactive
        pivstrat.idcs[i] = Int(idcs[i])
    end
    fill!(view(pivstrat.h, 1:nactive), zero(F))
    return nothing
end

"""
leja2_init!(pivstrat::GeoPivStratFunctor, nextidx::Int, nactive::Int=length(pivstrat.h))

Initialize minimum-distance vector `h` from pivot `nextidx`.

# Arguments

  - `pivstrat::GeoPivStratFunctor`: Functor with distance vector to initialize
  - `nextidx::Int`: Global index of selected pivot
  - `nactive::Int`: Number of active entries in `idcs`/`h`
"""
function leja2_init!(
    pivstrat::GeoPivStratFunctor, nextidx::Int, nactive::Int=length(pivstrat.h)
)
    pos = _positions(pivstrat)
    @inbounds for i in 1:nactive
        pivstrat.h[i] = norm(pos[pivstrat.idcs[i]] - pos[nextidx])
    end
    return nothing
end

"""
leja2!(pivstrat::GeoPivStratFunctor, nextidx::Int, nactive::Int=length(pivstrat.h))

Update minimum-distance vector `h` after selecting pivot `nextidx`.

# Arguments

  - `pivstrat::GeoPivStratFunctor`: Functor with distance vector to update
  - `nextidx::Int`: Global index of selected pivot
  - `nactive::Int`: Number of active entries in `idcs`/`h`
"""
function leja2!(pivstrat::GeoPivStratFunctor, nextidx::Int, nactive::Int=length(pivstrat.h))
    pos = _positions(pivstrat)
    @inbounds for i in 1:nactive
        d = norm(pos[pivstrat.idcs[i]] - pos[nextidx])
        if d < pivstrat.h[i]
            pivstrat.h[i] = d
        end
    end
    return nothing
end

"""
    (pivstrat::Leja2Functor{D,F})(::AbstractArray)

Select the next pivot with maximum minimum distance to selected points.

Chooses the point that is farthest from the set of already selected points,
then updates the distance vector for subsequent iterations.

# Arguments

  - `::AbstractArray`: Row/column data (unused, selection is purely geometric)

# Returns

  - `nextidx::Int`: Index of the point with maximum distance to selected points
"""
function (pivstrat::Leja2Functor{D,F})(::AbstractArray) where {D,F}
    nactive = pivstrat.nactive
    nextidx = argmax(view(pivstrat.h, 1:nactive))
    if all(iszero, view(pivstrat.h, 1:nactive))
        leja2_init!(pivstrat, pivstrat.idcs[nextidx], nactive)
    else
        leja2!(pivstrat, pivstrat.idcs[nextidx], nactive)
    end

    return nextidx
end
