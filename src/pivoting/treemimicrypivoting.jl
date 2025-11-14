
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

function (pivstrat::TreeMimicryPivoting{D,T})(
    F::V, refidcs::V, maxrank::Int
) where {D,T,V<:Vector{Int}}
    c = sum(pivstrat.refpos[refidcs]) ./ length(refidcs)
    usedidcs = zeros(Int, maxrank)
    return TreeMimicryPivotingFunctor{D,T}(F, c, pivstrat.tree, pivstrat.pos, usedidcs)
end

center(tree::T, node::Int) where {T} = error("Not implemented for type $T")
values(tree::T, node::Int) where {T} = error("Not implemented for type $T")
children(tree::T, node::Int) where {T} = error("Not implemented for type $T")
firstchild(tree::T, node::Int) where {T} = error("Not implemented for type $T")

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

function (pivstrat::TreeMimicryPivotingFunctor{D,F})() where {D,F<:Real}
    nodeidcs = values(pivstrat.tree, findcluster(pivstrat, pivstrat.F))
    w = zeros(F, length(nodeidcs))
    for (idx, node) in enumerate(nodeidcs)
        w[idx] = 1 / norm(pivstrat.pos[node] - pivstrat.c)
    end
    pivstrat.usedidcs[1] = nodeidcs[argmax(w)]

    return pivstrat.usedidcs[1]
end

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
