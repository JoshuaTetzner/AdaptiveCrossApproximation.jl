# Direction-aware filter for TreeMimicryPivoting. Groups basis functions by
# orientation (edge / normal direction) and rotates through the groups in phases:
# a phase stays active until its convergence criterion (a PhaseExtrapolator) fires,
# then the next unfinished direction is selected in round-robin order. Basis
# functions carrying an as-yet-unused edge direction are preferred, and the score
# of an index is down-weighted by how many pivots already share its edge direction.

"""
    DirectionFilter

Direction/orientation filter for [`TreeMimicryPivoting`](@ref). Restricts descent and
pivot selection to basis functions of the currently active direction, cycling through
directions in phases. Built through [`TreeMimicryPivoting2`](@ref).

# Fields

  - `edgedirections::Vector{Int32}`: per-index edge-direction group id
  - `basisdirections::Vector{Int32}`: per-index normal-direction group id
  - `nodedirectionids::Vector{Vector{Int32}}`: dominant direction ids per tree node
"""
struct DirectionFilter
    edgedirections::Vector{Int32}
    basisdirections::Vector{Int32}
    nodedirectionids::Vector{Vector{Int32}}
end

"""
    DirectionFilterState{CC}

Per-factorization state of a [`DirectionFilter`](@ref): the shared convergence
criterion plus the set of directions whose phase has finished.
"""
mutable struct DirectionFilterState{CC}
    convcrit::CC
    edgedirections::Vector{Int32}
    basisdirections::Vector{Int32}
    nodedirectionids::Vector{Vector{Int32}}
    finisheddirections::Vector{Int32}
    nfinished::Int
end

"""
    TreeMimicryPivoting2(refpos, pos, edgedirections, basisdirections, nodedirectionids, tree)

Tree-mimicry pivoting with a [`DirectionFilter`](@ref): pivots mimic the reference
distribution while cycling through orientation groups in phases.
"""
function TreeMimicryPivoting2(
    refpos::Vector{SVector{D,T}},
    pos::Vector{SVector{D,T}},
    edgedirections::Vector{Int32},
    basisdirections::Vector{Int32},
    nodedirectionids::Vector{Vector{Int32}},
    tree,
) where {D,T<:Real}
    filter = DirectionFilter(edgedirections, basisdirections, nodedirectionids)
    return TreeMimicryPivoting{D,T,typeof(tree),typeof(filter)}(refpos, pos, tree, filter)
end

function (pivstrat::TreeMimicryPivoting{D,T,TreeType,<:DirectionFilter})(
    convcrit::PhaseExtrapolatorFunctor,
    refidcs::AbstractVector{Int},
    idcs::AbstractVector{Int},
    maxrank::Int,
) where {D,T,TreeType}
    f = pivstrat.filter
    filterstate = DirectionFilterState(
        convcrit, f.edgedirections, f.basisdirections, f.nodedirectionids, Int32[], 0
    )
    return _functor(pivstrat, filterstate, refidcs, idcs, maxrank)
end

_buildpivstrat(
    convcrit,
    strat::TreeMimicryPivoting{D,T,TreeType,<:DirectionFilter},
    refidcs,
    idcs,
    maxrank,
) where {D,T,TreeType} = strat(convcrit, refidcs, idcs, maxrank)

# --- filter interface implementation ---------------------------------------
function _reset_filterstate!(fs::DirectionFilterState)
    empty!(fs.finisheddirections)
    fs.nfinished = 0
    return nothing
end

@inline function _contains(xs::AbstractVector{Int32}, x::Int32)
    @inbounds for y in xs
        y == x && return true
    end
    return false
end

@inline _matches_direction(dir::Int32, basisdir::Int32) = dir < 0 || basisdir == dir

@inline function _accepts_index(
    fs::DirectionFilterState, ::TreeMimicryPivotingFunctor, idx::Int, dir::Int32
)
    return _matches_direction(dir, fs.basisdirections[idx])
end

# Non-allocating subtree traversal. `values(tree, node)` on an internal node
# concatenates the whole subtree's indices into a fresh array; on a leaf it
# returns the stored index vector by reference. So we descend via the abstract
# tree interface (firstchild/children) and only call `values` on leaves. This
# keeps the same work but removes the per-check subtree materialization that
# dominated far-assembly allocations.
function _subtree_has_zerodir(tree, node::Int, fs::DirectionFilterState)
    if iszero(firstchild(tree, node))
        @inbounds for idx in values(tree, node)
            iszero(fs.basisdirections[idx]) && return true
        end
        return false
    end
    for child in children(tree, node)
        _subtree_has_zerodir(tree, Int(child), fs) && return true
    end
    return false
end

function _subtree_has_unused_match(
    functor::TreeMimicryPivotingFunctor,
    tree,
    node::Int,
    fs::DirectionFilterState,
    dir::Int32,
    nused::Int,
)
    if iszero(firstchild(tree, node))
        @inbounds for idx in values(tree, node)
            _used_index(functor, idx, nused) && continue
            _matches_direction(dir, fs.basisdirections[idx]) && return true
        end
        return false
    end
    for child in children(tree, node)
        _subtree_has_unused_match(functor, tree, Int(child), fs, dir, nused) && return true
    end
    return false
end

function _subtree_has_unused_dir(
    functor::TreeMimicryPivotingFunctor,
    tree,
    node::Int,
    fs::DirectionFilterState,
    dir::Int32,
    nused::Int,
)
    if iszero(firstchild(tree, node))
        @inbounds for idx in values(tree, node)
            _used_index(functor, idx, nused) && continue
            fs.basisdirections[idx] == dir && return true
        end
        return false
    end
    for child in children(tree, node)
        _subtree_has_unused_dir(functor, tree, Int(child), fs, dir, nused) && return true
    end
    return false
end

function _accepts_node(
    fs::DirectionFilterState, functor::TreeMimicryPivotingFunctor, node::Int, dir::Int32
)
    dir < 0 && return true
    if iszero(dir)
        return _subtree_has_zerodir(functor.pivoting.tree, node, fs)
    end
    return _contains(fs.nodedirectionids[node], dir)
end

@inline function _edge_count(
    fs::DirectionFilterState, functor::TreeMimicryPivotingFunctor, idx::Int, nused::Int
)
    dir = fs.edgedirections[idx]
    count = 0
    @inbounds for k in 1:nused
        count += fs.edgedirections[functor.usedidcs[k]] == dir
    end
    return count
end

@inline _reset_phase!(fs::DirectionFilterState, dir::Int32) = reset_phase!(fs.convcrit, dir)

# --- direction phasing -----------------------------------------------------
@inline function _finished_direction(fs::DirectionFilterState, dir::Int32)
    @inbounds for i in 1:(fs.nfinished)
        fs.finisheddirections[i] == dir && return true
    end
    return false
end

function _mark_finished_direction!(fs::DirectionFilterState, dir::Int32)
    dir <= 0 && return nothing
    _finished_direction(fs, dir) && return nothing
    push!(fs.finisheddirections, dir)
    fs.nfinished += 1
    return nothing
end

function _has_unused_basis_direction(
    fs::DirectionFilterState, functor::TreeMimicryPivotingFunctor, dir::Int32, nused::Int
)
    tree = functor.pivoting.tree
    @inbounds for i in 1:(functor.nactive)
        node = functor.farfield[i]
        _emptycluster(functor, node) && continue
        _subtree_has_unused_match(functor, tree, node, fs, dir, nused) && return true
    end
    return false
end

function _available_direction(
    fs::DirectionFilterState, functor::TreeMimicryPivotingFunctor, dir::Int32, nused::Int
)
    tree = functor.pivoting.tree
    @inbounds for i in 1:(functor.nactive)
        node = functor.farfield[i]
        _emptycluster(functor, node) && continue
        _accepts_node(fs, functor, node, dir) || continue
        _subtree_has_unused_dir(functor, tree, node, fs, dir, nused) && return true
    end
    return false
end

function _next_unfinished_direction(
    fs::DirectionFilterState,
    functor::TreeMimicryPivotingFunctor,
    nused::Int,
    lastdir::Int32,
)
    bestafter = Int32(0)
    bestwrap = Int32(0)
    @inbounds for i in 1:(functor.nactive)
        node = functor.farfield[i]
        _emptycluster(functor, node) && continue
        for dir in fs.nodedirectionids[node]
            _finished_direction(fs, dir) && continue
            _available_direction(fs, functor, dir, nused) || continue
            if dir > lastdir
                (iszero(bestafter) || dir < bestafter) && (bestafter = dir)
            else
                (iszero(bestwrap) || dir < bestwrap) && (bestwrap = dir)
            end
        end
    end
    return iszero(bestafter) ? bestwrap : bestafter
end

function _all_dominant_directions_finished(
    fs::DirectionFilterState, functor::TreeMimicryPivotingFunctor, nused::Int
)
    @inbounds for i in 1:(functor.nactive)
        node = functor.farfield[i]
        _emptycluster(functor, node) && continue
        for dir in fs.nodedirectionids[node]
            _finished_direction(fs, dir) && continue
            _available_direction(fs, functor, dir, nused) && return false
        end
    end
    return true
end

function _initial_direction(
    fs::DirectionFilterState, functor::TreeMimicryPivotingFunctor, nused::Int
)
    lastdir = fs.convcrit.currentdirection
    if iszero(nused)
        dir = _next_unfinished_direction(fs, functor, nused, Int32(0))
        !iszero(dir) && return dir
        _has_unused_basis_direction(fs, functor, Int32(0), nused) && return Int32(0)
        return _nodirection()
    end

    if fs.convcrit.lastconverged
        _mark_finished_direction!(fs, lastdir)
        _all_dominant_directions_finished(fs, functor, nused) ||
            (fs.convcrit.lastconverged = false)
    end

    nextdir = _next_unfinished_direction(fs, functor, nused, lastdir)
    !iszero(nextdir) && return nextdir
    _has_unused_basis_direction(fs, functor, Int32(0), nused) && return Int32(0)
    return _nodirection()
end

function _advance_direction(
    fs::DirectionFilterState, functor::TreeMimicryPivotingFunctor, nused::Int, dir::Int32
)
    nextdir = _next_unfinished_direction(fs, functor, nused, dir)
    if !iszero(nextdir) && nextdir != dir
        return (nextdir, true)
    elseif !iszero(dir) && _has_unused_basis_direction(fs, functor, Int32(0), nused)
        return (Int32(0), true)
    elseif dir != _nodirection()
        return (_nodirection(), true)
    end
    return (dir, false)
end
