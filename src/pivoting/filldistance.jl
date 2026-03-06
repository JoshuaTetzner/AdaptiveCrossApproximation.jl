
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

  - `h::Vector{F}`: Current minimum distance from each point to selected points
  - `pos::Vector{SVector{D,F}}`: Geometric positions corresponding to indices
"""
mutable struct FillDistanceFunctor{D,F<:Real} <: GeoPivStratFunctor
    h::Vector{F}
    idcs::Vector{Int}
    nactive::Int
    pos::Vector{SVector{D,F}}
end

"""
    (pivstrat::FillDistance{D,F})(idcs::AbstractArray{Int})

Create a `FillDistanceFunctor` for the given index subset.

Initializes the functor with positions corresponding to `idcs`, preparing it for
pivot selection within the submatrix.

# Arguments

  - `idcs::AbstractArray{Int}`: Indices of points to consider

# Returns

  - `FillDistanceFunctor`: Initialized functor with distance tracking
"""
function (pivstrat::FillDistance{D,F})(idcs::AbstractArray{Int}) where {D,F}
    nactive = length(idcs)
    return FillDistanceFunctor(zeros(F, nactive), collect(Int, idcs), nactive, pivstrat.pos)
end

function Base.resize!(
    pivstrat::FillDistanceFunctor{D,F}, nactive::Integer
) where {D,F<:Real}
    nactive < 0 && throw(ArgumentError("nactive must be non-negative"))
    resize!(pivstrat.h, nactive)
    resize!(pivstrat.idcs, nactive)
    pivstrat.nactive = min(pivstrat.nactive, Int(nactive))
    return pivstrat
end

function reset!(
    pivstrat::FillDistanceFunctor{D,F}, idcs::AbstractVector{<:Integer}
) where {D,F<:Real}
    nactive = length(idcs)
    length(pivstrat.h) < nactive && resize!(pivstrat, nactive)
    pivstrat.nactive = nactive

    @inbounds for i in 1:nactive
        pivstrat.idcs[i] = Int(idcs[i])
    end
    fill!(view(pivstrat.h, 1:nactive), zero(F))
    return pivstrat
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
    all(iszero, view(pivstrat.h, 1:nactive)) && (return pivstrat())
    nextidx = argmax(view(pivstrat.h, 1:nactive))
    maxval = pivstrat.h[nextidx]

    for k in 1:nactive
        pivstrat.h[k] == 0.0 && continue
        newfd = zero(F)
        for ind in 1:nactive
            d = norm(pivstrat.pos[pivstrat.idcs[k]] - pivstrat.pos[pivstrat.idcs[ind]])
            if pivstrat.h[ind] > d
                newfd < d && (newfd = d)
            else
                newfd < pivstrat.h[ind] && (newfd = pivstrat.h[ind])
            end
        end
        newfd <= maxval && (nextidx = k; maxval = newfd)
    end

    AdaptiveCrossApproximation.leja2!(pivstrat, pivstrat.idcs[nextidx], nactive)

    return nextidx
end
