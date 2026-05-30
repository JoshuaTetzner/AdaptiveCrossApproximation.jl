struct TreeMimicryPivoting2{D,T,TreeType} <: GeoPivStrat
    refpos::Vector{SVector{D,T}}
    pos::Vector{SVector{D,T}}
    edgedirections::Vector{Int32}
    basisdirections::Vector{Int32}
    nodedirectionids::Vector{Vector{Int32}}
    tree::TreeType
end

function TreeMimicryPivoting2(
    refpos::Vector{SVector{D,T}},
    pos::Vector{SVector{D,T}},
    edgedirections::Vector{Int32},
    basisdirections::Vector{Int32},
    nodedirectionids::Vector{Vector{Int32}},
    tree,
) where {D,T<:Real}
    return TreeMimicryPivoting2{D,T,typeof(tree)}(
        refpos, pos, edgedirections, basisdirections, nodedirectionids, tree
    )
end

mutable struct TreeMimicryPivotingFunctor2{D,T,TreeType} <: GeoPivStratFunctor
    pivoting::TreeMimicryPivoting2{D,T,TreeType}
    convcrit::OversampIFNormEstFunctor{T}
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
    finisheddirections::Vector{Int32}
    nfinished::Int
end

function (pivstrat::TreeMimicryPivoting2{D,T})(
    convcrit::OversampIFNormEstFunctor{T},
    refidcs::AbstractVector{Int},
    idcs::AbstractVector{Int},
    maxrank::Int,
) where {D,T}
    nactive = length(idcs)
    return TreeMimicryPivotingFunctor2(
        pivstrat,
        convcrit,
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
        Int32[],
        0,
    )
end

_buildpivstrat(convcrit, strat::TreeMimicryPivoting2, refidcs, idcs, maxrank) =
    strat(convcrit, refidcs, idcs, maxrank)

function Base.resize!(pivstrat::TreeMimicryPivotingFunctor2, nactive::Int)
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
    pivstrat::TreeMimicryPivotingFunctor2{D,T},
    refidcs::AbstractVector{Int},
    idcs::AbstractVector{Int},
) where {D,T}
    resize!(pivstrat, length(idcs))
    @inbounds for i in 1:(pivstrat.nactive)
        pivstrat.farfield[i] = idcs[i]
    end
    pivstrat.refcentroid = _centroid(pivstrat.pivoting.refpos, refidcs)
    fill!(view(pivstrat.h, 1:(pivstrat.nactive)), zero(T))
    fill!(view(pivstrat.leja, 1:(pivstrat.nactive)), one(T))
    fill!(view(pivstrat.w, 1:(pivstrat.nactive)), zero(T))
    fill!(pivstrat.emptyclusters, 0)
    fill!(pivstrat.usednodes, 0)
    fill!(pivstrat.usedidcs, 0)
    empty!(pivstrat.finisheddirections)
    pivstrat.nempty = 0
    pivstrat.nfinished = 0
    return nothing
end

@inline function local_reset!(
    pivstrat::TreeMimicryPivotingFunctor2{D,T}, n::Int
) where {D,T}
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

@inline function _contains(xs::AbstractVector{Int32}, x::Int32)
    @inbounds for y in xs
        y == x && return true
    end
    return false
end

@inline function _used_index(pivstrat::TreeMimicryPivotingFunctor2, idx::Int, nused::Int)
    @inbounds for k in 1:nused
        pivstrat.usedidcs[k] == idx && return true
    end
    return false
end

@inline _nodirection() = Int32(-1)
@inline _matches_direction(dir::Int32, basisdir::Int32) = dir < 0 || basisdir == dir

@inline function _used_basis_direction(
    pivstrat::TreeMimicryPivotingFunctor2, dir::Int32, nused::Int
)
    basisdirs = pivstrat.pivoting.basisdirections
    @inbounds for k in 1:nused
        basisdirs[pivstrat.usedidcs[k]] == dir && return true
    end
    return false
end

@inline function _finished_direction(pivstrat::TreeMimicryPivotingFunctor2, dir::Int32)
    @inbounds for i in 1:(pivstrat.nfinished)
        pivstrat.finisheddirections[i] == dir && return true
    end
    return false
end

function _mark_finished_direction!(pivstrat::TreeMimicryPivotingFunctor2, dir::Int32)
    dir <= 0 && return nothing
    _finished_direction(pivstrat, dir) && return nothing
    push!(pivstrat.finisheddirections, dir)
    pivstrat.nfinished += 1
    return nothing
end

@inline function _used_edge_count(
    pivstrat::TreeMimicryPivotingFunctor2, dir::Int32, nused::Int
)
    count = 0
    edgedirs = pivstrat.pivoting.edgedirections
    @inbounds for k in 1:nused
        count += edgedirs[pivstrat.usedidcs[k]] == dir
    end
    return count
end

@inline function _emptycluster(pivstrat::TreeMimicryPivotingFunctor2, node::Int)
    @inbounds for i in 1:(pivstrat.nempty)
        pivstrat.emptyclusters[i] == node && return true
    end
    return false
end

@inline function _mark_emptycluster!(pivstrat::TreeMimicryPivotingFunctor2, node::Int)
    _emptycluster(pivstrat, node) && return nothing
    if pivstrat.nempty == length(pivstrat.emptyclusters)
        resize!(pivstrat.emptyclusters, max(1, 2 * length(pivstrat.emptyclusters)))
    end
    pivstrat.nempty += 1
    pivstrat.emptyclusters[pivstrat.nempty] = node
    return nothing
end

@inline function _node_has_direction(
    pivstrat::TreeMimicryPivotingFunctor2, node::Int, dir::Int32
)
    dir < 0 && return true
    if iszero(dir)
        basisdirs = pivstrat.pivoting.basisdirections
        @inbounds for idx in values(pivstrat.pivoting.tree, node)
            basisdirs[idx] == dir && return true
        end
        return false
    end
    return _contains(pivstrat.pivoting.nodedirectionids[node], dir)
end

function _has_unused_basis_direction(
    pivstrat::TreeMimicryPivotingFunctor2, dir::Int32, nused::Int
)
    basisdirs = pivstrat.pivoting.basisdirections
    tree = pivstrat.pivoting.tree
    @inbounds for i in 1:(pivstrat.nactive)
        node = pivstrat.farfield[i]
        _emptycluster(pivstrat, node) && continue
        for idx in values(tree, node)
            _used_index(pivstrat, idx, nused) && continue
            _matches_direction(dir, basisdirs[idx]) && return true
        end
    end
    return false
end

@inline function _phase_converged(pivstrat::TreeMimicryPivotingFunctor2)
    return pivstrat.convcrit.lastconverged
end

function _available_direction(pivstrat::TreeMimicryPivotingFunctor2, dir::Int32, nused::Int)
    tree = pivstrat.pivoting.tree
    basisdirs = pivstrat.pivoting.basisdirections
    @inbounds for i in 1:(pivstrat.nactive)
        node = pivstrat.farfield[i]
        _emptycluster(pivstrat, node) && continue
        _node_has_direction(pivstrat, node, dir) || continue
        for idx in values(tree, node)
            _used_index(pivstrat, idx, nused) && continue
            basisdirs[idx] == dir && return true
        end
    end
    return false
end

function _next_unfinished_direction(
    pivstrat::TreeMimicryPivotingFunctor2, nused::Int, lastdir::Int32
)
    bestafter = Int32(0)
    bestwrap = Int32(0)

    @inbounds for i in 1:(pivstrat.nactive)
        node = pivstrat.farfield[i]
        _emptycluster(pivstrat, node) && continue

        for dir in pivstrat.pivoting.nodedirectionids[node]
            _finished_direction(pivstrat, dir) && continue
            _available_direction(pivstrat, dir, nused) || continue
            if dir > lastdir
                (iszero(bestafter) || dir < bestafter) && (bestafter = dir)
            else
                (iszero(bestwrap) || dir < bestwrap) && (bestwrap = dir)
            end
        end
    end

    !iszero(bestafter) && return bestafter
    return bestwrap
end

function _all_dominant_directions_finished(
    pivstrat::TreeMimicryPivotingFunctor2, nused::Int
)
    @inbounds for i in 1:(pivstrat.nactive)
        node = pivstrat.farfield[i]
        _emptycluster(pivstrat, node) && continue

        for dir in pivstrat.pivoting.nodedirectionids[node]
            _finished_direction(pivstrat, dir) && continue
            _available_direction(pivstrat, dir, nused) && return false
        end
    end
    return true
end

function _active_direction(pivstrat::TreeMimicryPivotingFunctor2, nused::Int)
    lastdir = pivstrat.convcrit.currentdirection
    if iszero(nused)
        dir = _next_unfinished_direction(pivstrat, nused, Int32(0))
        !iszero(dir) && return dir
        _has_unused_basis_direction(pivstrat, Int32(0), nused) && return Int32(0)
        return _nodirection()
    end

    if _phase_converged(pivstrat)
        _mark_finished_direction!(pivstrat, lastdir)
        _all_dominant_directions_finished(pivstrat, nused) ||
            (pivstrat.convcrit.lastconverged = false)
    end

    nextdir = _next_unfinished_direction(pivstrat, nused, lastdir)
    !iszero(nextdir) && return nextdir
    _has_unused_basis_direction(pivstrat, Int32(0), nused) && return Int32(0)
    return _nodirection()
end

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
    pivstrat::TreeMimicryPivotingFunctor2{D,T}, nodes, npivot::Int
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

function _best_node(pivstrat::TreeMimicryPivotingFunctor2, nodes, npivot::Int)
    n = _node_score!(pivstrat, nodes, npivot)
    return nodes[if npivot == 1
        _first_index(pivstrat.w, n)
    else
        bestindex(pivstrat.leja, pivstrat.h, pivstrat.w, n, npivot)
    end]
end

function _candidate_nodes(pivstrat::TreeMimicryPivotingFunctor2, nodes, dir::Int32)
    out = Int[]
    Base.haslength(nodes) && sizehint!(out, length(nodes))
    @inbounds for node in nodes
        inode = Int(node)
        if !_emptycluster(pivstrat, inode) && _node_has_direction(pivstrat, inode, dir)
            push!(out, inode)
        end
    end
    return out
end

function _findcluster(pivstrat::TreeMimicryPivotingFunctor2, nodes, npivot::Int, dir::Int32)
    candidates = _candidate_nodes(pivstrat, nodes, dir)
    while !isempty(candidates)
        node = _best_node(pivstrat, candidates, npivot)
        if iszero(firstchild(pivstrat.pivoting.tree, node))
            nodeidcs = values(pivstrat.pivoting.tree, node)
            allused = true
            @inbounds for idx in nodeidcs
                _used_index(pivstrat, idx, npivot - 1) && continue
                allused = false
                if _matches_direction(dir, pivstrat.pivoting.basisdirections[idx])
                    return node
                end
            end
            allused && _mark_emptycluster!(pivstrat, node)
            filter!(!=(node), candidates)
            continue
        end

        target = _findcluster(pivstrat, children(pivstrat.pivoting.tree, node), npivot, dir)
        !iszero(target) && return target

        filter!(!=(node), candidates)
    end
    return 0
end

function _leaf_score(
    pivstrat::TreeMimicryPivotingFunctor2{D,T}, localidx::Int, basisidx::Int, npivot::Int
) where {D,T}
    if npivot == 1
        return pivstrat.w[localidx]^4
    end
    exponent = T(2) / T(npivot - 1)
    edgecount = _used_edge_count(
        pivstrat, pivstrat.pivoting.edgedirections[basisidx], npivot - 1
    )
    edgeweight = iszero(edgecount) ? T(1) : inv(one(T) + T(edgecount))
    return (pivstrat.leja[localidx]^exponent) *
           pivstrat.h[localidx] *
           (pivstrat.w[localidx]^4) *
           edgeweight
end

function _best_leaf_index(
    pivstrat::TreeMimicryPivotingFunctor2, nodeidcs, npivot::Int, dir::Int32
)
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

    hasnewedge = false
    @inbounds for i in 1:n
        idx = nodeidcs[i]
        _used_index(pivstrat, idx, npivot - 1) && continue
        !_matches_direction(dir, pivstrat.pivoting.basisdirections[idx]) && continue
        if iszero(
            _used_edge_count(pivstrat, pivstrat.pivoting.edgedirections[idx], npivot - 1)
        )
            hasnewedge = true
            break
        end
    end

    best = 0
    bestscore = zero(eltype(pivstrat.w))
    @inbounds for i in 1:n
        idx = nodeidcs[i]
        _used_index(pivstrat, idx, npivot - 1) && continue
        !_matches_direction(dir, pivstrat.pivoting.basisdirections[idx]) && continue
        hasnewedge &&
            !iszero(
                _used_edge_count(
                    pivstrat, pivstrat.pivoting.edgedirections[idx], npivot - 1
                ),
            ) &&
            continue
        score = _leaf_score(pivstrat, i, idx, npivot)
        if iszero(best) || score > bestscore
            best = i
            bestscore = score
        end
    end

    return best
end

function (pivstrat::TreeMimicryPivotingFunctor2)(npivot::Int)
    direction = _active_direction(pivstrat, npivot - 1)
    #println(direction)
    reset_phase!(pivstrat.convcrit, direction, npivot)
    while true
        target = _findcluster(
            pivstrat, view(pivstrat.farfield, 1:(pivstrat.nactive)), npivot, direction
        )
        if iszero(target)
            nextdir = _next_unfinished_direction(pivstrat, npivot - 1, direction)
            if !iszero(nextdir) && nextdir != direction
                direction = nextdir
                reset_phase!(pivstrat.convcrit, direction, npivot)
                continue
            elseif !iszero(direction) &&
                _has_unused_basis_direction(pivstrat, Int32(0), npivot - 1)
                direction = Int32(0)
                reset_phase!(pivstrat.convcrit, direction, npivot)
                continue
            elseif direction != _nodirection()
                direction = _nodirection()
                reset_phase!(pivstrat.convcrit, direction, npivot)
                continue
            end
            throw(ArgumentError("No unused tree mimicry pivot candidate found."))
        end
        nodeidcs = values(pivstrat.pivoting.tree, target)
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

(pivstrat::TreeMimicryPivotingFunctor2)() = pivstrat(1)
