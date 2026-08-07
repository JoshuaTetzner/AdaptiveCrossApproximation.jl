"""
    TreeMimicryPivoting{D,T,TreeType,Filter} <: GeoPivStrat

Tree-aware mimicry pivoting strategy.

This strategy adapts the `MimicryPivoting` idea to a hierarchical tree of clusters.
Instead of selecting individual points directly, it navigates the tree to pick
clusters and then nodes within those clusters so that the selected pivots mimic a
reference distribution at multiple scales.

A pluggable `Filter` restricts the tree descent and the pivot selection. The default
[`NoFilter`](@ref) accepts every node and index, recovering the plain geometric
strategy. [`TreeMimicryPivoting2`](@ref) plugs in a direction filter that groups
basis functions by orientation and rotates through them in phases.

# Fields

  - `refpos::Vector{SVector{D,T}}`: Reference positions to mimic (e.g., parent pivots)
  - `pos::Vector{SVector{D,T}}`: Candidate point positions
  - `tree::TreeType`: Tree providing cluster centers, children and values
  - `filter::Filter`: Restriction on descent and pivot selection

# Type parameters

  - `D`: spatial dimension
  - `T`: numeric type for coordinates
  - `TreeType`: tree adapter (must implement `center`, `values`, `children`, `firstchild`)
  - `Filter`: filter type, statically dispatched in the per-pivot hot loops
"""
struct TreeMimicryPivoting{D,T,TreeType,Filter} <: GeoPivStrat
    refpos::Vector{SVector{D,T}}
    pos::Vector{SVector{D,T}}
    tree::TreeType
    filter::Filter
end

"""
    NoFilter

Identity pivot filter: accepts every node and index, assigns unit weight and runs a
single (directionless) phase. Recovers the plain tree-mimicry pivoting behaviour.
"""
struct NoFilter end

"""
    NoFilterState

Per-factorization state of [`NoFilter`](@ref); stateless.
"""
struct NoFilterState end

"""
    TreeMimicryPivoting(refpos, pos, tree)

Build a [`TreeMimicryPivoting`](@ref) strategy with the default [`NoFilter`](@ref),
i.e. plain tree-aware mimicry pivoting with no directional restriction.

# Arguments

  - `refpos::Vector{SVector{D,T}}`: Reference positions to mimic (e.g., parent pivots).
  - `pos::Vector{SVector{D,T}}`: Candidate point positions.
  - `tree`: Tree adapter providing `center`, `values`, `children` and `firstchild`
    (the ACAH2Trees extension supplies a ready-made adapter for `H2Trees.TwoNTree`).
"""
function TreeMimicryPivoting(
    refpos::Vector{SVector{D,T}}, pos::Vector{SVector{D,T}}, tree
) where {D,T<:Real}
    return TreeMimicryPivoting{D,T,typeof(tree),NoFilter}(refpos, pos, tree, NoFilter())
end

mutable struct TreeMimicryPivotingFunctor{D,T,TreeType,Filter,FState} <: GeoPivStratFunctor
    pivoting::TreeMimicryPivoting{D,T,TreeType,Filter}
    filterstate::FState
    nactive::Int
    refcentroid::SVector{D,T}
    farfield::Vector{Int}
    h::Vector{T}
    leja::Vector{T}
    w::Vector{T}
    emptyclusters::Vector{Int}
    nempty::Int
    usednodes::Vector{Int}
    usedidcs::Vector{Int}
end

function _functor(
    pivstrat::TreeMimicryPivoting{D,T}, filterstate, refidcs, idcs, maxrank::Int
) where {D,T}
    nactive = length(idcs)
    return TreeMimicryPivotingFunctor(
        pivstrat,
        filterstate,
        nactive,
        _centroid(pivstrat.refpos, refidcs),
        collect(Int, idcs),
        zeros(T, nactive),
        ones(T, nactive),
        zeros(T, nactive),
        zeros(Int, maxrank),
        0,
        zeros(Int, maxrank),
        zeros(Int, maxrank),
    )
end

function (pivstrat::TreeMimicryPivoting{D,T,TreeType,NoFilter})(
    refidcs::AbstractVector{Int}, idcs::AbstractVector{Int}, maxrank::Int
) where {D,T,TreeType}
    return _functor(pivstrat, NoFilterState(), refidcs, idcs, maxrank)
end

_buildpivstrat(
    strat::TreeMimicryPivoting{D,T,TreeType,NoFilter}, refidcs, idcs, maxrank
) where {D,T,TreeType} = strat(refidcs, idcs, maxrank)

function Base.resize!(pivstrat::TreeMimicryPivotingFunctor, nactive::Int)
    length(pivstrat.farfield) < nactive && resize!(pivstrat.farfield, nactive)
    if length(pivstrat.h) < nactive
        resize!(pivstrat.h, nactive)
        resize!(pivstrat.leja, nactive)
        resize!(pivstrat.w, nactive)
    end
    pivstrat.nactive = nactive
    return nothing
end

function reset!(
    pivstrat::TreeMimicryPivotingFunctor{D,T},
    refidcs::AbstractVector{Int},
    idcs::AbstractVector{Int},
) where {D,T}
    resize!(pivstrat, length(idcs))
    @inbounds for i in 1:(pivstrat.nactive)
        pivstrat.farfield[i] = Int(idcs[i])
    end
    pivstrat.refcentroid = _centroid(pivstrat.pivoting.refpos, refidcs)
    fill!(view(pivstrat.h, 1:(pivstrat.nactive)), zero(T))
    fill!(view(pivstrat.leja, 1:(pivstrat.nactive)), one(T))
    fill!(view(pivstrat.w, 1:(pivstrat.nactive)), zero(T))
    fill!(pivstrat.emptyclusters, 0)
    fill!(pivstrat.usednodes, 0)
    fill!(pivstrat.usedidcs, 0)
    pivstrat.nempty = 0
    _reset_filterstate!(pivstrat.filterstate)
    return nothing
end

@inline function local_reset!(pivstrat::TreeMimicryPivotingFunctor{D,T}, n::Int) where {D,T}
    if length(pivstrat.h) < n
        resize!(pivstrat.h, n)
        resize!(pivstrat.leja, n)
        resize!(pivstrat.w, n)
    end
    fill!(view(pivstrat.h, 1:n), zero(T))
    fill!(view(pivstrat.leja, 1:n), one(T))
    fill!(view(pivstrat.w, 1:n), zero(T))
    return n
end

# The package expects the `tree` object to implement these functions. Adaptors
# for concrete tree types should provide implementations in user code (the
# ACAH2Trees extension implements them for H2Trees.TwoNTree).

"""
    center(tree, node::Int)

Return the spatial centroid of `node` in `tree`.

Part of the tree adapter interface required by [`TreeMimicryPivoting`](@ref);
implement this for a custom tree type.
"""
center(tree::T, node::Int) where {T} = error("Not implemented for type $T")

"""
    values(tree, node)

Return the point indices contained in `node` (or each node in a vector of nodes) of `tree`.

Part of the tree adapter interface required by [`TreeMimicryPivoting`](@ref);
implement this for a custom tree type.
"""
values(tree::T, node::Union{Int,Vector{Int}}) where {T} =
    error("Not implemented for type $T")

"""
    children(tree, node::Int)

Return the child node indices of `node` in `tree`.

Part of the tree adapter interface required by [`TreeMimicryPivoting`](@ref);
implement this for a custom tree type.
"""
children(tree::T, node::Int) where {T} = error("Not implemented for type $T")

"""
    parent(tree, node::Int)

Return the parent node index of `node` in `tree`.

Part of the tree adapter interface required by [`TreeMimicryPivoting`](@ref);
implement this for a custom tree type.
"""
parent(tree::T, node::Int) where {T} = error("Not implemented for type $T")

"""
    firstchild(tree, node::Int)

Return the index of the first child of `node` in `tree`.

Part of the tree adapter interface required by [`TreeMimicryPivoting`](@ref);
implement this for a custom tree type.
"""
firstchild(tree::T, node::Int) where {T} = error("Not implemented for type $T")

# --- empty-cluster bookkeeping ---------------------------------------------
@inline function _emptycluster(pivstrat::TreeMimicryPivotingFunctor, node::Int)
    @inbounds for i in 1:(pivstrat.nempty)
        pivstrat.emptyclusters[i] == node && return true
    end
    return false
end

@inline function _mark_emptycluster!(pivstrat::TreeMimicryPivotingFunctor, node::Int)
    _emptycluster(pivstrat, node) && return nothing
    if pivstrat.nempty == length(pivstrat.emptyclusters)
        resize!(pivstrat.emptyclusters, max(1, 2 * length(pivstrat.emptyclusters)))
    end
    pivstrat.nempty += 1
    pivstrat.emptyclusters[pivstrat.nempty] = node
    return nothing
end

@inline function _used_index(pivstrat::TreeMimicryPivotingFunctor, idx::Int, nused::Int)
    @inbounds for k in 1:nused
        pivstrat.usedidcs[k] == idx && return true
    end
    return false
end

@inline _nodirection() = Int32(-1)

# --- filter interface (NoFilter defaults; DirectionFilter overrides) --------
# The engine is TreeMimicryPivoting2's machinery routed through these hooks.
# For NoFilter every hook collapses to the plain geometric behaviour: all nodes
# and indices are accepted, no edge weighting is applied, and a single
# directionless phase runs until the candidates are exhausted.
@inline _reset_filterstate!(::NoFilterState) = nothing
@inline _accepts_node(::NoFilterState, ::TreeMimicryPivotingFunctor, ::Int, ::Int32) = true
@inline _accepts_index(::NoFilterState, ::TreeMimicryPivotingFunctor, ::Int, ::Int32) = true
@inline _edge_count(::NoFilterState, ::TreeMimicryPivotingFunctor, ::Int, ::Int) = 0
@inline _reset_phase!(::NoFilterState, ::Int32) = nothing
@inline _initial_direction(::NoFilterState, ::TreeMimicryPivotingFunctor, ::Int) =
    _nodirection()
@inline _advance_direction(::NoFilterState, ::TreeMimicryPivotingFunctor, ::Int, ::Int32) =
    (_nodirection(), false)

# --- cluster scoring / descent ---------------------------------------------
@inline function _first_index(w, n)
    best = 1
    bestscore = w[1]^4
    @inbounds for i in 2:n
        score = w[i]^4
        if score > bestscore
            best = i
            bestscore = score
        end
    end
    return best
end

function _node_score!(
    pivstrat::TreeMimicryPivotingFunctor{D,T}, nodes, npivot::Int
) where {D,T}
    n = local_reset!(pivstrat, length(nodes))
    pos = pivstrat.pivoting.pos
    tree = pivstrat.pivoting.tree
    @inbounds for i in 1:n
        c = center(tree, nodes[i])
        pivstrat.w[i] = inv(norm(c - pivstrat.refcentroid))
        if npivot > 1
            pivstrat.h[i] = norm(pos[pivstrat.usedidcs[1]] - c)
            pivstrat.leja[i] = pivstrat.h[i]
            for j in 2:(npivot - 1)
                dist = norm(pos[pivstrat.usedidcs[j]] - c)
                pivstrat.h[i] = min(pivstrat.h[i], dist)
                pivstrat.leja[i] *= dist
            end
        end
    end
    return n
end

function _best_node(pivstrat::TreeMimicryPivotingFunctor, nodes, npivot::Int)
    n = _node_score!(pivstrat, nodes, npivot)
    return nodes[if npivot == 1
        _first_index(pivstrat.w, n)
    else
        bestindex(pivstrat.leja, pivstrat.h, pivstrat.w, n, npivot)
    end]
end

function _candidate_nodes(pivstrat::TreeMimicryPivotingFunctor, nodes, dir::Int32)
    fs = pivstrat.filterstate
    out = Int[]
    Base.haslength(nodes) && sizehint!(out, length(nodes))
    @inbounds for node in nodes
        inode = Int(node)
        if !_emptycluster(pivstrat, inode) && _accepts_node(fs, pivstrat, inode, dir)
            push!(out, inode)
        end
    end
    return out
end

function _findcluster(pivstrat::TreeMimicryPivotingFunctor, nodes, npivot::Int, dir::Int32)
    fs = pivstrat.filterstate
    tree = pivstrat.pivoting.tree
    candidates = _candidate_nodes(pivstrat, nodes, dir)
    while !isempty(candidates)
        node = _best_node(pivstrat, candidates, npivot)
        if iszero(firstchild(tree, node))
            nodeidcs = values(tree, node)
            allused = true
            @inbounds for idx in nodeidcs
                _used_index(pivstrat, idx, npivot - 1) && continue
                allused = false
                _accepts_index(fs, pivstrat, idx, dir) && return node
            end
            allused && _mark_emptycluster!(pivstrat, node)
            filter!(!=(node), candidates)
            continue
        end

        target = _findcluster(pivstrat, children(tree, node), npivot, dir)
        !iszero(target) && return target
        filter!(!=(node), candidates)
    end
    return 0
end

# --- leaf selection --------------------------------------------------------
function _leaf_score(
    pivstrat::TreeMimicryPivotingFunctor{D,T}, localidx::Int, basisidx::Int, npivot::Int
) where {D,T}
    npivot == 1 && return pivstrat.w[localidx]^4
    edgecount = _edge_count(pivstrat.filterstate, pivstrat, basisidx, npivot - 1)
    edgeweight = iszero(edgecount) ? T(1) : inv(one(T) + T(edgecount))
    exponent = T(2) / T(npivot - 1)
    return (pivstrat.leja[localidx]^exponent) *
           pivstrat.h[localidx] *
           (pivstrat.w[localidx]^4) *
           edgeweight
end

function _best_leaf_index(
    pivstrat::TreeMimicryPivotingFunctor, nodeidcs, npivot::Int, dir::Int32
)
    fs = pivstrat.filterstate
    n = local_reset!(pivstrat, length(nodeidcs))
    pos = pivstrat.pivoting.pos
    @inbounds for i in 1:n
        idx = nodeidcs[i]
        pivstrat.w[i] = inv(norm(pos[idx] - pivstrat.refcentroid))
        if npivot > 1
            pivstrat.h[i] = norm(pos[pivstrat.usedidcs[1]] - pos[idx])
            pivstrat.leja[i] = pivstrat.h[i]
            for j in 2:(npivot - 1)
                dist = norm(pos[pivstrat.usedidcs[j]] - pos[idx])
                pivstrat.h[i] = min(pivstrat.h[i], dist)
                pivstrat.leja[i] *= dist
            end
        end
    end

    # prefer indices carrying an as-yet-unused edge direction (no-op for NoFilter)
    hasnewedge = false
    @inbounds for i in 1:n
        idx = nodeidcs[i]
        _used_index(pivstrat, idx, npivot - 1) && continue
        _accepts_index(fs, pivstrat, idx, dir) || continue
        if iszero(_edge_count(fs, pivstrat, idx, npivot - 1))
            hasnewedge = true
            break
        end
    end

    best = 0
    bestscore = zero(eltype(pivstrat.w))
    @inbounds for i in 1:n
        idx = nodeidcs[i]
        _used_index(pivstrat, idx, npivot - 1) && continue
        _accepts_index(fs, pivstrat, idx, dir) || continue
        hasnewedge && !iszero(_edge_count(fs, pivstrat, idx, npivot - 1)) && continue
        score = _leaf_score(pivstrat, i, idx, npivot)
        if iszero(best) || score > bestscore
            best = i
            bestscore = score
        end
    end
    return best
end

# --- pivot selection -------------------------------------------------------
function (pivstrat::TreeMimicryPivotingFunctor)(npivot::Int)
    fs = pivstrat.filterstate
    tree = pivstrat.pivoting.tree
    direction = _initial_direction(fs, pivstrat, npivot - 1)
    _reset_phase!(fs, direction)
    while true
        target = _findcluster(
            pivstrat, view(pivstrat.farfield, 1:(pivstrat.nactive)), npivot, direction
        )
        if iszero(target)
            newdir, ok = _advance_direction(fs, pivstrat, npivot - 1, direction)
            ok || throw(ArgumentError("No unused tree mimicry pivot candidate found."))
            direction = newdir
            _reset_phase!(fs, direction)
            continue
        end
        nodeidcs = values(tree, target)
        leafidx = _best_leaf_index(pivstrat, nodeidcs, npivot, direction)
        if !iszero(leafidx)
            pivstrat.usednodes[npivot] = target
            pivstrat.usedidcs[npivot] = nodeidcs[leafidx]
            issubset(nodeidcs, view(pivstrat.usedidcs, 1:npivot)) &&
                _mark_emptycluster!(pivstrat, target)
            return pivstrat.usedidcs[npivot]
        end
        issubset(nodeidcs, view(pivstrat.usedidcs, 1:(npivot - 1))) &&
            _mark_emptycluster!(pivstrat, target)
    end
end

(pivstrat::TreeMimicryPivotingFunctor)() = pivstrat(1)

function update_refcentroid!(
    functor::TreeMimicryPivotingFunctor, refidcs::AbstractVector{<:Integer}
)
    functor.refcentroid = _centroid(functor.pivoting.refpos, refidcs)
    return functor
end
