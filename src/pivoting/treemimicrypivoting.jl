"""
    TreeMimicryPivoting{D,T,TreeType} <: GeoPivStrat

Tree-aware mimicry pivoting strategy.

This strategy adapts the `MimicryPivoting` idea to a hierarchical tree of
clusters. Instead of selecting individual points directly, it navigates the
tree to pick clusters and then nodes within those clusters so that the selected
pivots mimic a reference distribution at multiple scales.

# Fields

  - `refpos::Vector{SVector{D,T}}`: Reference positions to mimic (e.g., parent pivots)
  - `pos::Vector{SVector{D,T}}`: Candidate point positions
  - `tree::TreeType`: Tree structure providing cluster centers, children and values

# Type parameters

  - `D`: spatial dimension
  - `T`: numeric type for coordinates
  - `TreeType`: type of the tree adapter (must implement `center`, `values`, `children`, `firstchild`)
"""
struct TreeMimicryPivoting{D,T,TreeType} <: GeoPivStrat
    refpos::Vector{SVector{D,T}}
    pos::Vector{SVector{D,T}}
    tree::TreeType

    function TreeMimicryPivoting{D,T}(refpos, pos, tree) where {D,T}
        return new{D,T,typeof(tree)}(refpos, pos, tree)
    end
end

function TreeMimicryPivoting(
    refpos::Vector{SVector{D,T}}, pos::Vector{SVector{D,T}}, tree
) where {D,T<:Real}
    return TreeMimicryPivoting{D,T}(refpos, pos, tree)
end

mutable struct TreeMimicryPivotingFunctor{D,T,TreeType} <: GeoPivStratFunctor
    pivoting::TreeMimicryPivoting{D,T,TreeType}
    nactive::Int
    refcentroid::SVector{D,T}
    farfield::Vector{Int}
    h::Vector{T}
    leja::Vector{T}
    w::Vector{T}
    emptyclusters::Vector{Int}
    nempty::Int
    usedidcs::Vector{Int}
end

function (pivstrat::TreeMimicryPivoting{D,T})(
    refidcs::AbstractVector{Int}, idcs::AbstractVector{Int}, maxrank::Int
) where {D,T}
    refcentroid = _centroid(pivstrat.refpos, refidcs)
    farfieldbuf = collect(Int, idcs)
    farfieldlen = length(farfieldbuf)
    h = zeros(T, farfieldlen)
    leja = ones(T, farfieldlen)
    w = zeros(T, farfieldlen)
    usedidcs = zeros(Int, maxrank)
    emptyclusters = zeros(Int, maxrank)
    return TreeMimicryPivotingFunctor(
        pivstrat,
        farfieldlen,
        refcentroid,
        farfieldbuf,
        h,
        leja,
        w,
        emptyclusters,
        0,
        usedidcs,
    )
end

_buildpivstrat(strat::TreeMimicryPivoting, refidcs, idcs, maxrank) =
    strat(refidcs, idcs, maxrank)

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
) where {D,T<:Real}
    resize!(pivstrat, length(idcs))
    @inbounds for i in 1:(pivstrat.nactive)
        pivstrat.farfield[i] = Int(idcs[i])
    end
    pivstrat.refcentroid = _centroid(pivstrat.pivoting.refpos, refidcs)
    fill!(view(pivstrat.h, 1:(pivstrat.nactive)), zero(T))
    fill!(view(pivstrat.leja, 1:(pivstrat.nactive)), one(T))
    fill!(view(pivstrat.w, 1:(pivstrat.nactive)), zero(T))
    fill!(pivstrat.emptyclusters, 0)
    fill!(pivstrat.usedidcs, 0)
    pivstrat.nempty = 0
    return nothing
end

@inline function local_resize!(pivstrat::TreeMimicryPivotingFunctor, localnactive::Int)
    if length(pivstrat.h) < localnactive
        resize!(pivstrat.h, localnactive)
        resize!(pivstrat.leja, localnactive)
        resize!(pivstrat.w, localnactive)
    end
    return localnactive
end

@inline function local_reset!(
    pivstrat::TreeMimicryPivotingFunctor{D,T}, localidcs::AbstractVector{<:Integer}
) where {D,T<:Real}
    nlocal = local_resize!(pivstrat, length(localidcs))
    fill!(view(pivstrat.h, 1:nlocal), zero(T))
    fill!(view(pivstrat.leja, 1:nlocal), one(T))
    fill!(view(pivstrat.w, 1:nlocal), zero(T))
    return nlocal
end

#The package expects the `tree` object to implement these functions. Adaptors
#for concrete tree types should provide implementations in user code.
center(tree::T, node::Int) where {T} = error("Not implemented for type $T")
values(tree::T, node::Union{Int,Vector{Int}}) where {T} =
    error("Not implemented for type $T")
children(tree::T, node::Int) where {T} = error("Not implemented for type $T")
parent(tree::T, node::Int) where {T} = error("Not implemented for type $T")
firstchild(tree::T, node::Int) where {T} = error("Not implemented for type $T")

@inline function _is_emptycluster(pivstrat::TreeMimicryPivotingFunctor, node::Int)
    @inbounds for i in 1:(pivstrat.nempty)
        pivstrat.emptyclusters[i] == node && return true
    end
    return false
end

@inline function _mark_emptycluster!(pivstrat::TreeMimicryPivotingFunctor, node::Int)
    _is_emptycluster(pivstrat, node) && return pivstrat
    if pivstrat.nempty >= length(pivstrat.emptyclusters)
        throw(
            ArgumentError(
                "Too many empty clusters tracked ($(pivstrat.nempty + 1)) for allocated capacity $(length(pivstrat.emptyclusters)). Increase maxrank.",
            ),
        )
    end
    pivstrat.nempty += 1
    pivstrat.emptyclusters[pivstrat.nempty] = node
    return pivstrat
end

@inline function _filter_emptyclusters(
    pivstrat::TreeMimicryPivotingFunctor, nodes::AbstractVector{Int}
)
    pivstrat.nempty == 0 && return nodes
    filtered = Int[]
    sizehint!(filtered, length(nodes))
    @inbounds for node in nodes
        !_is_emptycluster(pivstrat, node) && push!(filtered, node)
    end
    return filtered
end

@inline function _filter_emptyclusters(pivstrat::TreeMimicryPivotingFunctor, nodes)
    if pivstrat.nempty == 0
        return collect(Int, nodes)
    end

    filtered = Int[]
    Base.haslength(nodes) && sizehint!(filtered, length(nodes))
    @inbounds for node in nodes
        inode = Int(node)
        !_is_emptycluster(pivstrat, inode) && push!(filtered, inode)
    end
    return filtered
end

function findcluster(
    pivstrat::TreeMimicryPivotingFunctor{D,T}, nodes::AbstractVector{Int}
) where {D,T<:Real}
    nlocal = local_reset!(pivstrat, nodes)
    tree = pivstrat.pivoting.tree
    @inbounds for idx in 1:nlocal
        pivstrat.w[idx] = 1 / norm(center(tree, nodes[idx]) - pivstrat.refcentroid)
    end
    w = view(pivstrat.w, 1:nlocal)
    imax = argmax(w)
    node = nodes[imax]
    iszero(firstchild(tree, node)) && return node
    return findcluster(pivstrat, collect(Int, children(tree, node)))
end

function findcluster(
    pivstrat::TreeMimicryPivotingFunctor{D,T}, idcs::AbstractVector{Int}, npivot::Int
) where {D,T<:Real}
    nlocal = local_reset!(pivstrat, idcs)
    pos = pivstrat.pivoting.pos
    tree = pivstrat.pivoting.tree

    @inbounds for i in 1:nlocal
        pivstrat.w[i] = 1 / norm(center(tree, idcs[i]) - pivstrat.refcentroid)
        pivstrat.h[i] = norm(pos[pivstrat.usedidcs[1]] - center(tree, idcs[i]))
        pivstrat.leja[i] = pivstrat.h[i]
        @inbounds for j in 2:(npivot - 1)
            dist = norm(pos[pivstrat.usedidcs[j]] - center(tree, idcs[i]))
            if dist < pivstrat.h[i]
                pivstrat.h[i] = dist
            end
            pivstrat.leja[i] *= dist
        end
    end
    node = idcs[bestindex(pivstrat.leja, pivstrat.h, pivstrat.w, nlocal, npivot)]

    # Might need rescue measure here!!!
    iszero(firstchild(tree, node)) && return node

    chds = _filter_emptyclusters(pivstrat, children(tree, node))
    if isempty(chds)
        _mark_emptycluster!(pivstrat, node)
        activefarfield = _filter_emptyclusters(
            pivstrat, view(pivstrat.farfield, 1:(pivstrat.nactive))
        )
        return findcluster(pivstrat, activefarfield, npivot)
    end
    return findcluster(pivstrat, chds, npivot)
end

function (pivstrat::TreeMimicryPivotingFunctor{D,F})() where {D,F<:Real}
    targetcluster = findcluster(pivstrat, view(pivstrat.farfield, 1:(pivstrat.nactive)))
    pos = pivstrat.pivoting.pos
    tree = pivstrat.pivoting.tree
    nodeidcs = values(tree, targetcluster)
    nlocal = local_reset!(pivstrat, nodeidcs)
    w = view(pivstrat.w, 1:nlocal)
    for (idx, node) in enumerate(nodeidcs)
        w[idx] = 1 / norm(pos[node] - pivstrat.refcentroid)
    end
    pivstrat.usedidcs[1] = nodeidcs[argmax(w)]
    issubset(nodeidcs, view(pivstrat.usedidcs, 1:1)) &&
        _mark_emptycluster!(pivstrat, targetcluster)

    return pivstrat.usedidcs[1]
end

function (pivstrat::TreeMimicryPivotingFunctor{D,F})(npivot::Int) where {D,F<:Real}
    activefarfield = _filter_emptyclusters(
        pivstrat, view(pivstrat.farfield, 1:(pivstrat.nactive))
    )
    pos = pivstrat.pivoting.pos
    tree = pivstrat.pivoting.tree
    targetcluster = findcluster(pivstrat, activefarfield, npivot)
    nodeidcs = values(tree, targetcluster)
    # might be a performance killer
    @assert !issubset(nodeidcs, view(pivstrat.usedidcs, 1:(npivot - 1)))

    nlocal = local_reset!(pivstrat, nodeidcs)
    @inbounds for idx in 1:nlocal
        pivstrat.w[idx] = 1 / norm(pos[nodeidcs[idx]] - pivstrat.refcentroid)
        pivstrat.h[idx] = norm(pos[pivstrat.usedidcs[1]] - pos[nodeidcs[idx]])
        pivstrat.leja[idx] = pivstrat.h[idx]
        @inbounds for j in 2:(npivot - 1)
            dist = norm(pos[pivstrat.usedidcs[j]] - pos[nodeidcs[idx]])
            if dist < pivstrat.h[idx]
                pivstrat.h[idx] = dist
            end
            pivstrat.leja[idx] *= dist
        end
    end

    pivstrat.usedidcs[npivot] = nodeidcs[bestindex(
        pivstrat.leja, pivstrat.h, pivstrat.w, nlocal, npivot
    )]
    issubset(nodeidcs, view(pivstrat.usedidcs, 1:npivot)) &&
        _mark_emptycluster!(pivstrat, targetcluster)
    return pivstrat.usedidcs[npivot]
end
