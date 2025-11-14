#=struct TreeMimicryPivoting{D,T<:Real} <: GeoPivStrat
    refpos::Vector{SVector{D,T}}
    pos::Vector{SVector{D,T}}
    tree::H2Trees.H2ClusterTree
end

mutable struct TreeMimicryPivotingFunctor{D,T<:Real} <: GeoPivStratFunctor
    F::Vector{Int}
    c::SVector{D,T}
    tree::H2Trees.H2ClusterTree
    pos::Vector{SVector{D,T}}
    usedidcs::Vector{Int}
end

function (pivstrat::TreeMimicryPivoting{D,T})(
    F::V, refidcs::V, maxrank::Int
) where {D,T,V<:Vector{Int}}
    c = sum(pivstrat.refpos[refidcs]) ./ length(refidcs)
    usedidcs = zeros(Int, maxrank)
    return TreeMimicryPivotingFunctor{D,T}(F, c, pivstrat.tree, pivstrat.pos, usedidcs)
end

function findcluster(
    pivstrat::TreeMimicryPivotingFunctor{D,T}, F::Vector{I}
) where {D,T<:Real,I}
    for (idx, f) in enumerate(F)
        w[idx] = 1 / norm(H2Trees.center(pivstrat.tree, f) - pivstrat.c)
    end
    iszero(H2Tree.firstchild(pivstrat.tree, F[argmax(w)])) && return F[argmax(w)]

    return findcluster(
        pivstrat, collect(H2Trees.children(pivstrat.tree, F[argmax(w)])), npivot
    )
end

function findcluster(
    pivstrat::TreeMimicryPivotingFunctor{D,T}, F::Vector{I}, npivot::I
) where {D,T<:Real,I}
    w = zeros(T, length(F))
    h = zeros(T, length(F))
    leja = ones(T, length(F))
    for (idx, f) in enumerate(F)
        w[idx] = 1 / norm(H2Trees.center(pivstrat.tree, f) - pivstrat.c)
        h[idx] = norm(pivstrat.pos[pivstrat.usedidcs[1]] - H2Trees.center(pivstrat.tree, f))
        leja[idx] *= norm(
            pivstrat.pos[pivstrat.usedidcs[1]] - H2Trees.center(pivstrat.tree, f)
        )
        for sidx in pivstrat.usedidcs[2:(npivot - 1)]
            if norm(pivstrat.pos[sidx] - H2Trees.center(pivstrat.tree, f)) < h[idx]
                h[idx] = norm(pivstrat.pos[sidx] - H2Trees.center(pivstrat.tree, f))
            end
            leja[idx] *= norm(pivstrat.pos[sidx] - H2Trees.center(pivstrat.tree, f))
        end
    end
    cluster = F[argmax(leja .^ (2 / (npivot - 1)) .* h .* w .^ 4)]
    iszero(H2Tree.firstchild(pivstrat.tree, cluster)) && return cluster

    return findcluster(pivstrat, collect(H2Trees.children(pivstrat.tree, cluster)), npivot)
end

function (pivstrat::TreeMimicryPivotingFunctor{D,F})() where {D,F<:Real}
    nodeidcs = H2Trees.values(pivstrat.tree, findcluster(pivstrat, pivstrat.F))
    w = zeros(T, length(F))
    for (idx, node) in enumerate(nodeidcs)
        w[idx] = 1 / norm(pivstrat.pos[node] - pivstrat.c)
    end
    pivstrat.usedidcs[1] = nodeidcs[argmax(w)]
    return nodeidcs[argmax(w)]
end

function (pivstrat::TreeMimicryPivotingFunctor{D,F})(npivot::Int) where {D,F<:Real}
    nodeidcs = H2Trees.values(pivstrat.tree, findcluster(pivstrat, pivstrat.F, npivot))

    w = zeros(T, length(F))
    h = zeros(T, length(F))
    leja = ones(T, length(F))
    for (idx, node) in enumerate(nodeidcs)
        w[idx] = 1 / norm(pivstrat.pos[node] - pivstrat.c)
        h[idx] = norm(pivstrat.pos[pivstrat.usedidcs[1]] - pivstrat.pos[node])
        leja[idx] *= norm(pivstrat.pos[pivstrat.usedidcs[1]] - pivstrat.pos[node])
        for sidx in pivstrat.usedidcs[2:(npivots - 1)]
            if norm(pivstrat.pos[sidx] - pivstrat.pos[node]) < h[idx]
                h[idx] = norm(pivstrat.pos[sidx] - pivstrat.pos[node])
            end
            leja[idx] *= norm(pivstrat.pos[sidx] - pivstrat.pos[node])
        end
    end
    usedidcs[npivot] = nodeidcs[argmax(leja .^ (2 / (npivot - 1)) .* h .* w .^ 4)]

    return findcluster(pivstrat, collect(H2Trees.children(pivstrat.tree, cluster)))
end=#

function AdaptiveCrossApproximation.center(tree::H2Trees.H2ClusterTree, node::Int)
    return H2Trees.center(tree, node)
end

function AdaptiveCrossApproximation.values(tree::H2Trees.H2ClusterTree, node::Int)
    return H2Trees.values(tree, node)
end

function AdaptiveCrossApproximation.children(tree::H2Trees.H2ClusterTree, node::Int)
    return H2Trees.children(tree, node)
end

function AdaptiveCrossApproximation.firstchild(tree::H2Trees.H2ClusterTree, node::Int)
    return H2Trees.firstchild(tree, node)
end
