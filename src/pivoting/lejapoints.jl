
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
mutable struct Leja2Functor{D,F<:Real} <: GeoPivStratFunctor
    h::Vector{F}
    idcs::Vector{Int}
    nactive::Int
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
    nactive = length(idcs)
    return Leja2Functor{D,F}(zeros(F, nactive), collect(Int, idcs), nactive, pivstrat.pos)
end

function Base.resize!(pivstrat::Leja2Functor{D,F}, nactive::Integer) where {D,F<:Real}
    nactive < 0 && throw(ArgumentError("nactive must be non-negative"))
    resize!(pivstrat.h, nactive)
    resize!(pivstrat.idcs, nactive)
    pivstrat.nactive = min(pivstrat.nactive, Int(nactive))
    return pivstrat
end

function reset!(
    pivstrat::Leja2Functor{D,F}, idcs::AbstractVector{<:Integer}
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
    @inbounds for i in 1:nactive
        pivstrat.h[i] = norm(pivstrat.pos[pivstrat.idcs[i]] - pivstrat.pos[nextidx])
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
    @inbounds for i in 1:nactive
        d = norm(pivstrat.pos[pivstrat.idcs[i]] - pivstrat.pos[nextidx])
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
