
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
struct FillDistanceFunctor{D,F<:Real} <: GeoPivStratFunctor
    h::Vector{F}
    idcs::Vector{Int}
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
    return FillDistanceFunctor(zeros(F, length(idcs)), idcs, pivstrat.pos)
end

"""
    (pivstrat::Union{Leja2Functor{D,F},FillDistanceFunctor{D,F}})()

Select the first point as the initial pivot.

Computes distances from all points to the first point and returns index 1.
"""
function (pivstrat::Union{Leja2Functor{D,F},FillDistanceFunctor{D,F}})() where {D,F}
    @views pivstrat.h .= norm.(
        pivstrat.pos[pivstrat.idcs] .- Scalar(pivstrat.pos[pivstrat.idcs[1]])
    )

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
    nextidx = argmax(pivstrat.h)
    maxval = pivstrat.h[nextidx]

    for k in eachindex(pivstrat.h)
        newfd = 0.0
        for (ind, pos) in enumerate(pivstrat.pos[pivstrat.idcs])
            if pivstrat.h[ind] > norm(pivstrat.pos[pivstrat.idcs[k]] - pos)
                newfd < norm(pivstrat.pos[pivstrat.idcs[k]] - pos) &&
                    (newfd = norm(pivstrat.pos[pivstrat.idcs[k]] - pos))
            else
                newfd < pivstrat.h[ind] && (newfd = pivstrat.h[ind])
            end
        end
        newfd < maxval && (nextidx=k; maxval=newfd)
    end

    AdaptiveCrossApproximation.leja2!(pivstrat, nextidx)

    return nextidx
end
