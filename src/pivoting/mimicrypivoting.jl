
mutable struct MimicryPivoting{D,F<:Real} <: GeoPivStrat
    refpos::Vector{SVector{D,F}}
    pos::Vector{SVector{D,F}}
end

mutable struct MimicryPivotingFunctor{D,F<:Real} <: GeoPivStratFunctor
    pos::Vector{SVector{D,F}}
    idcs::Vector{Int}
    h::Vector{F}
    leja::Vector{F}
    w::Vector{F}
end

function (strat::MimicryPivoting{D,F})(refidcs, rcidcs) where {D,F}
    ref = sum(strat.refpos[refidcs]) ./ length(refidcs)
    h = zeros(F, length(rcidcs))
    w = 1 ./ norm.(strat.pos[rcidcs] .- Scalar(ref))
    leja = ones(F, length(rcidcs))

    return MimicryPivotingFunctor{D,F}(strat.pos, rcidcs, h, leja, w)
end

function (strat::MimicryPivotingFunctor{D,F})() where {D,F<:Real}
    nextidx = strat.idcs[argmax(strat.w)]
    @views strat.h .= norm.(strat.pos[strat.idcs] .- Scalar(strat.pos[nextidx]))
    strat.leja .*= norm.(strat.pos[strat.idcs] .- Scalar(strat.pos[nextidx]))
    return nextidx
end

function (strat::MimicryPivotingFunctor{D,F})(npivot::Int) where {D,F<:Real}
    nextidx = strat.idcs[argmax(
        strat.leja .^ (2 / (npivot - 1)) .* strat.h .* strat.w .^ 4
    )]
    leja2!(strat, nextidx)
    strat.leja .*= norm.(strat.pos[strat.idcs] .- Scalar(strat.pos[nextidx]))
    return nextidx
end
