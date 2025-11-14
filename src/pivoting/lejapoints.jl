
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

  - `h::Vector{F}`: Current minimum distance from each point to selected points
  - `idcs::Vector{Int}`: Indices of points being considered for selection
  - `pos::Vector{SVector{D,F}}`: Geometric positions corresponding to indices
"""
struct Leja2Functor{D,F<:Real} <: GeoPivStratFunctor
    h::Vector{F}
    idcs::Vector{Int}
    pos::Vector{SVector{D,F}}
end

"""
    (pivstrat::Leja2{D,F})(idcs::AbstractArray{Int})

Create a `Leja2Functor` for the given index subset.

Initializes the functor with positions corresponding to `idcs`, preparing it for
pivot selection within the submatrix.

# Arguments

  - `idcs::AbstractArray{Int}`: Indices of points to consider

# Returns

  - `Leja2Functor`: Initialized functor with distance tracking
"""
function (pivstrat::Leja2{D,F})(idcs::AbstractArray{Int}) where {D,F}
    return Leja2Functor{D,F}(zeros(F, length(idcs)), idcs, pivstrat.pos)
end

"""
    leja2!(pivstrat::GeoPivStratFunctor, nextidx::Int)

Update minimum distances after selecting pivot `nextidx`.

Computes distances from all points to the newly selected pivot and updates the
minimum distance vector `h` by taking element-wise minimum with new distances.
This shared helper is used by both Leja2 and fill distance strategies.

# Arguments

  - `pivstrat::GeoPivStratFunctor`: Functor with distance vector to update
  - `nextidx::Int`: Index of newly selected pivot
"""
function leja2!(pivstrat::GeoPivStratFunctor, nextidx::Int)
    newh = norm.(pivstrat.pos[pivstrat.idcs] .- Scalar(pivstrat.pos[nextidx]))
    all(==(0.0), pivstrat.h) && (pivstrat.h .= newh)
    for idx in eachindex(pivstrat.h)
        pivstrat.h[idx] > newh[idx] && (pivstrat.h[idx] = newh[idx])
    end
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
    nextidx = argmax(pivstrat.h)
    leja2!(pivstrat, pivstrat.idcs[nextidx])

    return nextidx
end
