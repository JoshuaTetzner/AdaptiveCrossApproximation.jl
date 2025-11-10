
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

"""
    TreeMimicryPivoting(refpos, pos, tree)

Convenience constructor inferring tree type. `refpos` and `pos` must be
vectors of `SVector{D,T}` coordinates and `tree` must provide required methods.
"""
function TreeMimicryPivoting(
    refpos::Vector{SVector{D,T}}, pos::Vector{SVector{D,T}}, tree
) where {D,T<:Real}
    return TreeMimicryPivoting{D,T}(refpos, pos, tree)
end

"""
    TreeMimicryPivotingFunctor{D,T,TreeType} <: GeoPivStratFunctor

Functor storing state for tree-based mimicry pivoting.

# Fields

  - `F::Vector{Int}`: Candidate cluster node indices to search
  - `c::SVector{D,T}`: Reference centroid used to bias selection
  - `tree::TreeType`: Tree providing cluster geometry and membership
  - `pos::Vector{SVector{D,T}}`: Point coordinates
  - `usedidcs::Vector{Int}`: Selected global point indices (filled progressively)
"""
mutable struct TreeMimicryPivotingFunctor{D,T,TreeType} <: GeoPivStratFunctor
    F::Vector{Int}
    c::SVector{D,T}
    tree::TreeType
    pos::Vector{SVector{D,T}}
    usedidcs::Vector{Int}

    function TreeMimicryPivotingFunctor{D,T}(
        F::Vector{Int}, c, tree, pos, usedidcs::Vector{Int}
    ) where {D,T}
        return new{D,T,typeof(tree)}(F, c, tree, pos, usedidcs)
    end
end

"""
    (pivstrat::TreeMimicryPivoting)(F, refidcs, maxrank)

Initialize a tree-aware mimicry functor.

`F` is a vector of tree node candidates (e.g., root children). The function
computes the reference centroid from `refidcs` and allocates `usedidcs` of length
`maxrank` for storing selected point indices.
"""
function (pivstrat::TreeMimicryPivoting{D,T})(
    F::V, refidcs::V, maxrank::Int
) where {D,T,V<:Vector{Int}}
    c = sum(pivstrat.refpos[refidcs]) ./ length(refidcs)
    usedidcs = zeros(Int, maxrank)
    return TreeMimicryPivotingFunctor{D,T}(F, c, pivstrat.tree, pivstrat.pos, usedidcs)
end

#The package expects the `tree` object to implement these functions. Adaptors
#for concrete tree types should provide implementations in user code.
center(tree::T, node::Int) where {T} = error("Not implemented for type $T")
values(tree::T, node::Int) where {T} = error("Not implemented for type $T")
children(tree::T, node::Int) where {T} = error("Not implemented for type $T")
firstchild(tree::T, node::Int) where {T} = error("Not implemented for type $T")

"""
    findcluster(pivstrat, F)

Find a leaf cluster (node) that best matches the reference centroid.

Traverses the tree greedily by choosing child clusters whose centers are
closest (in weighted inverse-distance sense) to the reference centroid `pivstrat.c`.
Returns a node index whose `firstchild` is zero (leaf) or recurses into children.
"""
function findcluster(
    pivstrat::TreeMimicryPivotingFunctor{D,T}, F::Vector{I}
) where {D,T<:Real,I}
    w = zeros(T, length(F))
    for (idx, f) in enumerate(F)
        w[idx] = 1 / norm(center(pivstrat.tree, f) - pivstrat.c)
    end
    iszero(firstchild(pivstrat.tree, F[argmax(w)])) && return F[argmax(w)]

    return findcluster(pivstrat, collect(children(pivstrat.tree, F[argmax(w)])))
end

"""
    findcluster(pivstrat, F, npivot)

Cluster-based selection used during later pivot iterations.

For each candidate cluster `f` in `F`, compute a composite score combining Leja
products, fill distances and inverse-distance weights to the reference centroid;
select the cluster maximizing this score and recurse until a leaf is reached.
"""
function findcluster(
    pivstrat::TreeMimicryPivotingFunctor{D,T}, F::Vector{I}, npivot::I
) where {D,T<:Real,I}
    w = zeros(T, length(F))
    h = zeros(T, length(F))
    leja = ones(T, length(F))
    for (idx, f) in enumerate(F)
        w[idx] = 1 / norm(center(pivstrat.tree, f) - pivstrat.c)
        h[idx] = norm(pivstrat.pos[pivstrat.usedidcs[1]] - center(pivstrat.tree, f))
        leja[idx] *= norm(pivstrat.pos[pivstrat.usedidcs[1]] - center(pivstrat.tree, f))
        for sidx in pivstrat.usedidcs[2:(npivot - 1)]
            if norm(pivstrat.pos[sidx] - center(pivstrat.tree, f)) < h[idx]
                h[idx] = norm(pivstrat.pos[sidx] - center(pivstrat.tree, f))
            end
            leja[idx] *= norm(pivstrat.pos[sidx] - center(pivstrat.tree, f))
        end
    end
    cluster = F[argmax(leja .^ (2 / (npivot - 1)) .* h .* w .^ 4)]
    iszero(firstchild(pivstrat.tree, cluster)) && return cluster

    return findcluster(pivstrat, collect(children(pivstrat.tree, cluster)), npivot)
end

"""
    (pivstrat::TreeMimicryPivotingFunctor)()

Select the first pivot by locating a promising leaf cluster and choosing the
point within that cluster that is closest to the reference centroid.
"""
function (pivstrat::TreeMimicryPivotingFunctor{D,F})() where {D,F<:Real}
    nodeidcs = values(pivstrat.tree, findcluster(pivstrat, pivstrat.F))
    w = zeros(F, length(nodeidcs))
    for (idx, node) in enumerate(nodeidcs)
        w[idx] = 1 / norm(pivstrat.pos[node] - pivstrat.c)
    end
    pivstrat.usedidcs[1] = nodeidcs[argmax(w)]

    return pivstrat.usedidcs[1]
end

"""
    (pivstrat::TreeMimicryPivotingFunctor)(npivot)

Select subsequent pivots by finding a cluster and then selecting the best
point within that cluster based on mimicry pivoting strategy.
"""
function (pivstrat::TreeMimicryPivotingFunctor{D,F})(npivot::Int) where {D,F<:Real}
    nodeidcs = values(pivstrat.tree, findcluster(pivstrat, pivstrat.F, npivot))
    w = zeros(F, length(nodeidcs))
    h = zeros(F, length(nodeidcs))
    leja = ones(F, length(nodeidcs))
    for (idx, node) in enumerate(nodeidcs)
        w[idx] = 1 / norm(pivstrat.pos[node] - pivstrat.c)
        h[idx] = norm(pivstrat.pos[pivstrat.usedidcs[1]] - pivstrat.pos[node])
        leja[idx] *= norm(pivstrat.pos[pivstrat.usedidcs[1]] - pivstrat.pos[node])
        for sidx in pivstrat.usedidcs[2:(npivot - 1)]
            if norm(pivstrat.pos[sidx] - pivstrat.pos[node]) < h[idx]
                h[idx] = norm(pivstrat.pos[sidx] - pivstrat.pos[node])
            end
            leja[idx] *= norm(pivstrat.pos[sidx] - pivstrat.pos[node])
        end
    end
    pivstrat.usedidcs[npivot] = nodeidcs[argmax(leja .^ (2 / (npivot - 1)) .* h .* w .^ 4)]

    return pivstrat.usedidcs[npivot]
end
