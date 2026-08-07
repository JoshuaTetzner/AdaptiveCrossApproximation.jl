"""
    MFIETreeMimicryPivoting(refpos, pos, tree; planarity_atol, planarity_rtol)

Tree-mimicry pivoting for MFIE interactions.

Before selecting a pivot, the strategy checks whether the current reference cluster is
planar. If it is, candidate positions in the same plane are excluded from the pivot
set. For a non-planar or geometrically degenerate reference cluster, the strategy
falls back to ordinary tree-mimicry selection without plane filtering.

The planarity test and the same-plane test use
`planarity_atol + planarity_rtol * reference_scale`, where `reference_scale` is the
largest distance from a reference point to the reference centroid.

A call returns `0` when the supplied far-field nodes contain no eligible candidate.
"""
struct MFIETreeMimicryPivoting{T,TreeType} <: GeoPivStrat
    refpos::Vector{SVector{3,T}}
    pos::Vector{SVector{3,T}}
    tree::TreeType
    planarity_atol::T
    planarity_rtol::T
end

function MFIETreeMimicryPivoting(
    refpos::Vector{SVector{3,T}},
    pos::Vector{SVector{3,T}},
    tree;
    planarity_atol::Real=sqrt(eps(T)),
    planarity_rtol::Real=sqrt(eps(T)),
) where {T<:Real}
    atol = T(planarity_atol)
    rtol = T(planarity_rtol)
    atol >= zero(T) || throw(ArgumentError("planarity_atol must be nonnegative"))
    rtol >= zero(T) || throw(ArgumentError("planarity_rtol must be nonnegative"))
    return MFIETreeMimicryPivoting{T,typeof(tree)}(refpos, pos, tree, atol, rtol)
end

mutable struct MFIETreeMimicryPivotingFunctor{T,TreeType} <: GeoPivStratFunctor
    pivoting::MFIETreeMimicryPivoting{T,TreeType}
    nactive::Int
    refcentroid::SVector{3,T}
    refnormal::SVector{3,T}
    planethreshold::T
    referenceisplanar::Bool
    farfield::Vector{Int}
    usedidcs::Vector{Int}
end

function _reference_plane(
    pos::Vector{SVector{3,T}},
    idcs::AbstractVector{Int},
    atol::T,
    rtol::T,
) where {T<:Real}
    origin = _centroid(pos, idcs)
    zerovec = zero(SVector{3,T})
    length(idcs) >= 3 || return false, origin, zerovec, atol

    scale = zero(T)
    axis = zerovec
    @inbounds for idx in idcs
        offset = pos[idx] - origin
        distance = norm(offset)
        if distance > scale
            scale = distance
            axis = offset
        end
    end

    threshold = atol + rtol * scale
    scale > threshold || return false, origin, zerovec, threshold

    crossnorm = zero(T)
    normal = zerovec
    @inbounds for idx in idcs
        candidate = cross(axis, pos[idx] - origin)
        candidate_norm = norm(candidate)
        if candidate_norm > crossnorm
            crossnorm = candidate_norm
            normal = candidate
        end
    end

    # Collinear points do not determine a unique plane.
    crossnorm > threshold * scale || return false, origin, zerovec, threshold
    normal /= crossnorm

    @inbounds for idx in idcs
        abs(dot(pos[idx] - origin, normal)) <= threshold ||
            return false, origin, zerovec, threshold
    end
    return true, origin, normal, threshold
end

function (strategy::MFIETreeMimicryPivoting{T})(
    refidcs::AbstractVector{Int}, idcs::AbstractVector{Int}, maxrank::Int
) where {T}
    planar, origin, normal, threshold = _reference_plane(
        strategy.refpos,
        refidcs,
        strategy.planarity_atol,
        strategy.planarity_rtol,
    )
    nactive = length(idcs)
    return MFIETreeMimicryPivotingFunctor(
        strategy,
        nactive,
        origin,
        normal,
        threshold,
        planar,
        collect(Int, idcs),
        zeros(Int, maxrank),
    )
end

_buildpivstrat(strategy::MFIETreeMimicryPivoting, refidcs, idcs, maxrank) =
    strategy(refidcs, idcs, maxrank)

function Base.resize!(strategy::MFIETreeMimicryPivotingFunctor, nactive::Int)
    length(strategy.farfield) < nactive && resize!(strategy.farfield, nactive)
    strategy.nactive = nactive
    return nothing
end

function reset!(
    strategy::MFIETreeMimicryPivotingFunctor{T},
    refidcs::AbstractVector{Int},
    idcs::AbstractVector{Int},
) where {T}
    resize!(strategy, length(idcs))
    @inbounds for i in eachindex(idcs)
        strategy.farfield[i] = idcs[i]
    end
    planar, origin, normal, threshold = _reference_plane(
        strategy.pivoting.refpos,
        refidcs,
        strategy.pivoting.planarity_atol,
        strategy.pivoting.planarity_rtol,
    )
    strategy.refcentroid = origin
    strategy.refnormal = normal
    strategy.planethreshold = threshold
    strategy.referenceisplanar = planar
    fill!(strategy.usedidcs, 0)
    return nothing
end

@inline function _mfie_used_index(
    strategy::MFIETreeMimicryPivotingFunctor, idx::Int, nused::Int
)
    @inbounds for i in 1:nused
        strategy.usedidcs[i] == idx && return true
    end
    return false
end

@inline function _same_reference_plane(
    strategy::MFIETreeMimicryPivotingFunctor, idx::Int
)
    strategy.referenceisplanar || return false
    return abs(
        dot(
            strategy.pivoting.pos[idx] - strategy.refcentroid,
            strategy.refnormal,
        ),
    ) <= strategy.planethreshold
end

@inline function _mfie_eligible(
    strategy::MFIETreeMimicryPivotingFunctor, idx::Int, nused::Int
)
    return !_mfie_used_index(strategy, idx, nused) &&
           !_same_reference_plane(strategy, idx)
end

function _mfie_node_has_candidate(
    strategy::MFIETreeMimicryPivotingFunctor, node::Int, nused::Int
)
    @inbounds for idx in values(strategy.pivoting.tree, node)
        _mfie_eligible(strategy, idx, nused) && return true
    end
    return false
end

function _mfie_candidate_nodes(
    strategy::MFIETreeMimicryPivotingFunctor, nodes, nused::Int
)
    candidates = Int[]
    Base.haslength(nodes) && sizehint!(candidates, length(nodes))
    @inbounds for node in nodes
        inode = Int(node)
        _mfie_node_has_candidate(strategy, inode, nused) && push!(candidates, inode)
    end
    return candidates
end

@inline function _mfie_inverse_distance(a, b, ::Type{T}) where {T}
    return inv(max(norm(a - b), eps(T)))
end

function _mfie_node_score(
    strategy::MFIETreeMimicryPivotingFunctor{T}, node::Int, npivot::Int
) where {T}
    nodecenter = center(strategy.pivoting.tree, node)
    weight = _mfie_inverse_distance(nodecenter, strategy.refcentroid, T)
    npivot == 1 && return weight^4

    position = strategy.pivoting.pos
    distance = norm(position[strategy.usedidcs[1]] - nodecenter)
    filldistance = distance
    lejaproduct = distance
    @inbounds for i in 2:(npivot - 1)
        distance = norm(position[strategy.usedidcs[i]] - nodecenter)
        filldistance = min(filldistance, distance)
        lejaproduct *= distance
    end
    exponent = T(2) / T(npivot - 1)
    return lejaproduct^exponent * filldistance * weight^4
end

function _mfie_best_node(
    strategy::MFIETreeMimicryPivotingFunctor, nodes::Vector{Int}, npivot::Int
)
    best = first(nodes)
    bestscore = _mfie_node_score(strategy, best, npivot)
    @inbounds for i in 2:length(nodes)
        node = nodes[i]
        score = _mfie_node_score(strategy, node, npivot)
        if score > bestscore
            best = node
            bestscore = score
        end
    end
    return best
end

function _mfie_find_leaf(
    strategy::MFIETreeMimicryPivotingFunctor, nodes, npivot::Int
)
    candidates = _mfie_candidate_nodes(strategy, nodes, npivot - 1)
    while !isempty(candidates)
        node = _mfie_best_node(strategy, candidates, npivot)
        iszero(firstchild(strategy.pivoting.tree, node)) && return node
        leaf = _mfie_find_leaf(
            strategy, children(strategy.pivoting.tree, node), npivot
        )
        !iszero(leaf) && return leaf
        filter!(!=(node), candidates)
    end
    return 0
end

function _mfie_leaf_score(
    strategy::MFIETreeMimicryPivotingFunctor{T}, idx::Int, npivot::Int
) where {T}
    position = strategy.pivoting.pos
    weight = _mfie_inverse_distance(position[idx], strategy.refcentroid, T)
    npivot == 1 && return weight^4

    distance = norm(position[strategy.usedidcs[1]] - position[idx])
    filldistance = distance
    lejaproduct = distance
    @inbounds for i in 2:(npivot - 1)
        distance = norm(position[strategy.usedidcs[i]] - position[idx])
        filldistance = min(filldistance, distance)
        lejaproduct *= distance
    end
    exponent = T(2) / T(npivot - 1)
    return lejaproduct^exponent * filldistance * weight^4
end

function (strategy::MFIETreeMimicryPivotingFunctor{T})(npivot::Int) where {T}
    nodes = view(strategy.farfield, 1:(strategy.nactive))
    leaf = _mfie_find_leaf(strategy, nodes, npivot)
    iszero(leaf) && return 0

    best = 0
    bestscore = zero(T)
    @inbounds for idx in values(strategy.pivoting.tree, leaf)
        _mfie_eligible(strategy, idx, npivot - 1) || continue
        score = _mfie_leaf_score(strategy, idx, npivot)
        if iszero(best) || score > bestscore
            best = idx
            bestscore = score
        end
    end
    strategy.usedidcs[npivot] = best
    return best
end

(strategy::MFIETreeMimicryPivotingFunctor)() = strategy(1)
