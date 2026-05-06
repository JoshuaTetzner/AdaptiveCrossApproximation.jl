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
struct TreeMimicryPivoting2{D,T,TreeType} <: GeoPivStrat
    refpos::Vector{SVector{D,T}}
    pos::Vector{SVector{D,T}}
    edges::Vector{SVector{D,T}}
    orientations::Vector{SVector{D,T}}
    tree::TreeType

    function TreeMimicryPivoting2{D,T}(refpos, pos, edges, orientations, tree) where {D,T}
        return new{D,T,typeof(tree)}(refpos, pos, edges, orientations, tree)
    end
end

function TreeMimicryPivoting2(
    refpos::Vector{SVector{D,T}},
    pos::Vector{SVector{D,T}},
    edges::Vector{SVector{D,T}},
    orientations::Vector{SVector{D,T}},
    tree,
) where {D,T<:Real}
    return TreeMimicryPivoting2{D,T}(refpos, pos, edges, orientations, tree)
end

mutable struct TreeMimicryPivotingFunctor2{D,T,TreeType} <: GeoPivStratFunctor
    pivoting::TreeMimicryPivoting2{D,T,TreeType}
    nactive::Int
    refcentroid::SVector{D,T}
    farfield::Vector{Int}
    h::Vector{T}
    leja::Vector{T}
    w::Vector{T}
    w2::Vector{T}
    emptyclusters::Vector{Int}
    nempty::Int
    usednodes::Vector{Int}
    usedidcs::Vector{Int}
end

function (pivstrat::TreeMimicryPivoting2{D,T})(
    refidcs::AbstractVector{Int}, idcs::AbstractVector{Int}, maxrank::Int
) where {D,T}
    refcentroid = _centroid(pivstrat.refpos, refidcs)
    farfieldbuf = collect(Int, idcs)
    farfieldlen = length(farfieldbuf)
    h = zeros(T, farfieldlen)
    leja = ones(T, farfieldlen)
    w = zeros(T, farfieldlen)
    w2 = zeros(T, farfieldlen)
    usedidcs = zeros(Int, maxrank)
    usednodes = zeros(Int, maxrank)
    emptyclusters = zeros(Int, maxrank)
    return TreeMimicryPivotingFunctor2(
        pivstrat,
        farfieldlen,
        refcentroid,
        farfieldbuf,
        h,
        leja,
        w,
        w2,
        emptyclusters,
        0,
        usedidcs,
        usednodes,
    )
end

_buildpivstrat(strat::TreeMimicryPivoting2, refidcs, idcs, maxrank) =
    strat(refidcs, idcs, maxrank)

function Base.resize!(
    pivstrat::TreeMimicryPivotingFunctor2{D,T,TreeType}, nactive::Int
) where {D,T,TreeType}
    length(pivstrat.farfield) < nactive && resize!(pivstrat.farfield, nactive)
    if length(pivstrat.h) < nactive
        resize!(pivstrat.h, nactive)
        resize!(pivstrat.leja, nactive)
        resize!(pivstrat.w, nactive)
        resize!(pivstrat.w2, nactive)
    end
    pivstrat.nactive = nactive
    return nothing
end

function reset!(
    pivstrat::TreeMimicryPivotingFunctor2{D,T,TreeType},
    refidcs::AbstractVector{Int},
    idcs::AbstractVector{Int},
) where {D,T,TreeType}
    resize!(pivstrat, length(idcs))
    @inbounds for i in 1:((pivstrat.nactive))
        pivstrat.farfield[i] = Int(idcs[i])
    end
    pivstrat.refcentroid = _centroid(pivstrat.pivoting.refpos, refidcs)
    fill!(view(pivstrat.h, 1:(pivstrat.nactive)), zero(T))
    fill!(view(pivstrat.leja, 1:(pivstrat.nactive)), one(T))
    fill!(view(pivstrat.w, 1:(pivstrat.nactive)), zero(T))
    fill!(view(pivstrat.w2, 1:(pivstrat.nactive)), zero(T))
    fill!(pivstrat.emptyclusters, 0)
    fill!(pivstrat.usedidcs, 0)
    fill!(pivstrat.usednodes, 0)
    pivstrat.nempty = 0
    return nothing
end

@inline function local_resize!(pivstrat::TreeMimicryPivotingFunctor2, localnactive::Int)
    if length(pivstrat.h) < localnactive
        resize!(pivstrat.h, localnactive)
        resize!(pivstrat.leja, localnactive)
        resize!(pivstrat.w, localnactive)
        resize!(pivstrat.w2, localnactive)
    end
    return localnactive
end

@inline function local_reset!(
    pivstrat::TreeMimicryPivotingFunctor2{D,T,TreeType},
    localidcs::AbstractVector{<:Integer},
) where {D,T,TreeType}
    nlocal = local_resize!(pivstrat, length(localidcs))
    fill!(view(pivstrat.h, 1:nlocal), zero(T))
    fill!(view(pivstrat.leja, 1:nlocal), one(T))
    fill!(view(pivstrat.w, 1:nlocal), zero(T))
    fill!(view(pivstrat.w2, 1:nlocal), zero(T))
    return nlocal
end

@inline function _is_emptycluster(
    pivstrat::TreeMimicryPivotingFunctor2{D,T,TreeType}, node::Int
) where {D,T,TreeType}
    @inbounds for i in 1:(pivstrat.nempty)
        pivstrat.emptyclusters[i] == node && return true
    end
    return false
end

@inline function _mark_emptycluster!(
    pivstrat::TreeMimicryPivotingFunctor2{D,T,TreeType}, node::Int
) where {D,T,TreeType}
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
    pivstrat::TreeMimicryPivotingFunctor2{D,T,TreeType}, nodes
) where {D,T,TreeType}
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
    pivstrat::TreeMimicryPivotingFunctor2{D,T,TreeType}, nodes::AbstractVector{Int}
) where {D,T,TreeType}
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
    pivstrat::TreeMimicryPivotingFunctor2{D,T,TreeType},
    idcs::AbstractVector{Int},
    npivot::Int,
) where {D,T,TreeType}
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

function (pivstrat::TreeMimicryPivotingFunctor2{D,T,TreeType})() where {D,T,TreeType}
    targetcluster = findcluster(pivstrat, view(pivstrat.farfield, 1:(pivstrat.nactive)))
    pivstrat.usednodes[1] = targetcluster
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

@inline round1(x) = round(x; digits=1)
@inline round_orientation(x) = round(x)
@inline fixzero(x) = iszero(x) ? zero(x) : x

@inline function orientation_key(d)
    x = fixzero(round1(d[1]))
    y = fixzero(round1(d[2]))
    z = fixzero(round1(d[3]))

    if x < 0 || (x == 0 && y < 0) || (x == 0 && y == 0 && z < 0)
        x = fixzero(-x)
        y = fixzero(-y)
        z = fixzero(-z)
    end

    return (x, y, z)
end

#=function orientation_weights!(weights, keys, usedkeys)
    counts = Dict{typeof(keys[1]),Int}()
    for k in usedkeys
        counts[k] = get(counts, k, 0) + 1
    end
    for i in eachindex(keys)
        weights[i] = 1.0 / (1 + get(counts, keys[i], 0))
    end
end=#

function orientation_weights!(weights, keys, usedkeys)
    K = typeof(keys[1])

    used_counts = Dict{K,Int}()
    available_counts = Dict{K,Int}()

    for k in usedkeys
        used_counts[k] = get(used_counts, k, 0) + 1
    end

    for k in keys
        available_counts[k] = get(available_counts, k, 0) + 1
    end

    for i in eachindex(keys)
        k = keys[i]
        weights[i] = available_counts[k]^(1 / 4) / (1 + get(used_counts, k, 0))
    end

    return weights
end

@inline function bestindex(
    leja::AbstractVector{F},
    h::AbstractVector{F},
    w::AbstractVector{F},
    w2::AbstractVector{F},
    nactive::Int,
    npivot::Int,
) where {F<:Real}
    nactive > 0 || throw(ArgumentError("nactive must be positive."))
    npivot > 1 || throw(ArgumentError("npivot must be larger than 1."))

    exponent = F(2) / F(npivot - 1)
    @inbounds begin
        nextlocal = 1
        bestscore = (leja[1]^exponent) * h[1] * (w[1]^4) * w2[1]^2
        for i in 2:nactive
            score = (leja[i]^exponent) * h[i] * (w[i]^4) * w2[i]^2
            if score > bestscore
                bestscore = score
                nextlocal = i
            end
        end
        return nextlocal
    end
end

function findcluster2(
    pivstrat::TreeMimicryPivotingFunctor2{D,T}, idcs::AbstractVector{Int}, npivot::Int
) where {D,T<:Real}
    nlocal = local_reset!(pivstrat, idcs)
    pos = pivstrat.pivoting.pos
    tree = pivstrat.pivoting.tree

    orientations = pivstrat.pivoting.orientations[idcs]
    usedorientations = pivstrat.pivoting.orientations[pivstrat.usednodes[1:(npivot - 1)]]
    keys = [orientation_key(d) for d in orientations]
    #usedkeys = [orientation_key(d) for d in usedorientations]
    usedkeys = Vector{NTuple{3,Float64}}(undef, npivot - 1)
    for ind in eachindex(usedorientations)
        uo = usedorientations[ind]
        key = orientation_key(uo)
        node = pivstrat.usednodes[ind]
        while !(key in keys) && node != 0
            node = parent(tree, node)
            node == 0 && break
            key = orientation_key(pivstrat.pivoting.orientations[node])
        end
        usedkeys[ind] = key
    end
    orientation_weights!(view(pivstrat.w2, 1:nlocal), keys, usedkeys)
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
    #if npivot > 5
    #node = idcs[bestindex(
    #    pivstrat.leja, pivstrat.h, pivstrat.w, pivstrat.w2, nlocal, npivot
    #)]
    #else
    node = idcs[bestindex(
        pivstrat.leja, pivstrat.h, pivstrat.w, pivstrat.w2, nlocal, npivot
    )]
    #end
    # Might need rescue measure here!!!
    iszero(firstchild(tree, node)) && return node

    chds = _filter_emptyclusters(pivstrat, children(tree, node))
    if isempty(chds)
        _mark_emptycluster!(pivstrat, node)
        activefarfield = _filter_emptyclusters(
            pivstrat, view(pivstrat.farfield, 1:(pivstrat.nactive))
        )
        return findcluster2(pivstrat, activefarfield, npivot)
    end
    return findcluster2(pivstrat, chds, npivot)
end

function (pivstrat::TreeMimicryPivotingFunctor2{D,T,TreeType})(
    npivot::Int
) where {D,T,TreeType}
    activefarfield = _filter_emptyclusters(
        pivstrat, view(pivstrat.farfield, 1:(pivstrat.nactive))
    )
    pos = pivstrat.pivoting.pos
    tree = pivstrat.pivoting.tree
    targetcluster = findcluster2(pivstrat, activefarfield, npivot)
    pivstrat.usednodes[npivot] = targetcluster
    nodeidcs = values(tree, targetcluster)
    # might be a performance killer
    @assert !issubset(nodeidcs, view(pivstrat.usedidcs, 1:(npivot - 1)))
    edges = pivstrat.pivoting.edges[nodeidcs]
    sedges = pivstrat.pivoting.edges[pivstrat.usedidcs[1:(npivot - 1)]]
    keys = [orientation_key(d) for d in edges]
    usedkeys = [orientation_key(d) for d in sedges]
    nlocal = local_reset!(pivstrat, nodeidcs)
    orientation_weights!(view(pivstrat.w2, 1:nlocal), keys, usedkeys)
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
        pivstrat.leja, pivstrat.h, pivstrat.w, pivstrat.w2, nlocal, npivot
    )]
    issubset(nodeidcs, view(pivstrat.usedidcs, 1:npivot)) &&
        _mark_emptycluster!(pivstrat, targetcluster)
    return pivstrat.usedidcs[npivot]
end
