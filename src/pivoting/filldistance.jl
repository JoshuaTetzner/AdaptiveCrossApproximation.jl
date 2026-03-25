
"""
    FillDistance{D,F<:Real} <: GeoPivStrat

Geometric pivoting strategy based on fill distance minimization.

Selects pivots to minimize the fill distance, promoting well-distributed sampling in geometric space.

# Fields

  - `pos::Vector{SVector{D,F}}`: Geometric positions of all points (D-dimensional)

# Type Parameters

  - `D`: Spatial dimension
  - `F`: Floating point type for coordinates
"""
struct FillDistance{D,F<:Real} <: GeoPivStrat
    pos::Vector{SVector{D,F}}
end

"""
    FillDistanceFunctor{D,F<:Real} <: PivStratFunctor

Stateful functor for fill distance pivot selection.

Maintains the minimum distances from each point to the set of selected points, updating them
as new pivots are chosen.

# Fields

    - `pivoting::FillDistance{D,F}`: Immutable strategy carrying geometric positions
    - `nactive::Int`: Active prefix length in state vectors
    - `idcs::Vector{Int}`: Active indices into `pivoting.pos`
    - `h::Vector{F}`: Current minimum distance from each point to selected points
"""
mutable struct FillDistanceFunctor{D,F<:Real} <: GeoPivStratFunctor
    pivoting::FillDistance{D,F}
    nactive::Int
    idcs::Vector{Int}
    h::Vector{F}
end

function (pivstrat::FillDistance{D,F})(idcs::AbstractVector{<:Integer}) where {D,F<:Real}
    nactive = length(idcs)
    return FillDistanceFunctor(pivstrat, nactive, collect(Int, idcs), zeros(F, nactive))
end

function (pivstrat::FillDistance{D,F})(nidcs::Int) where {D,F<:Real}
    return FillDistanceFunctor(pivstrat, nidcs, zeros(Int, nidcs), zeros(F, nidcs))
end

function Base.resize!(pivstrat::FillDistanceFunctor{D,F}, nactive::Int) where {D,F<:Real}
    length(pivstrat.h) < nactive && resize!(pivstrat.h, nactive)
    length(pivstrat.idcs) < nactive && resize!(pivstrat.idcs, nactive)
    pivstrat.nactive = nactive
    return nothing
end

function reset!(
    pivstrat::FillDistanceFunctor{D,F}, idcs::AbstractVector{<:Integer}
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
    (pivstrat::Union{Leja2Functor{D,F},FillDistanceFunctor{D,F}})()

Select the first point as the initial pivot.

Computes distances from all points to the first point and returns index 1.
"""
function (pivstrat::Union{Leja2Functor{D,F},FillDistanceFunctor{D,F}})() where {D,F}
    AdaptiveCrossApproximation.leja2_init!(pivstrat, pivstrat.idcs[1], pivstrat.nactive)
    return 1
end

"""
    (pivstrat::FillDistanceFunctor{D,F})(::AbstractArray)

Select the next pivot minimizing the fill distance with respect to the selected points and
updates the distance vector `h` for subsequent iterations.

# Arguments

  - `::AbstractArray`: Row/column data (unused, selection is purely geometric)

# Returns

  - `nextidx::Int`: Index of the point maximizing fill distance
"""
function (pivstrat::FillDistanceFunctor{D,F})(::AbstractArray) where {D,F}
    nactive = pivstrat.nactive
    pos = pivstrat.pivoting.pos
    all(iszero, view(pivstrat.h, 1:nactive)) && (return pivstrat())
    nextidx = argmax(view(pivstrat.h, 1:nactive))
    maxval = pivstrat.h[nextidx]

    for k in 1:nactive
        pivstrat.h[k] == 0.0 && continue
        newfd = zero(F)
        for ind in 1:nactive
            d = norm(pos[pivstrat.idcs[k]] - pos[pivstrat.idcs[ind]])
            if pivstrat.h[ind] > d
                newfd < d && (newfd = d)
            else
                newfd < pivstrat.h[ind] && (newfd = pivstrat.h[ind])
            end
        end
        newfd <= maxval && (nextidx=k; maxval=newfd)
    end

    AdaptiveCrossApproximation.leja2!(pivstrat, pivstrat.idcs[nextidx], nactive)

    return nextidx
end
