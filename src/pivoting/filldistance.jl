
struct FillDistance{D,F<:Real} <: GeoPivStrat
    pos::Vector{SVector{D,F}}
end

struct FillDistanceFunctor{D,F<:Real} <: PivStratFunctor
    h::Vector{F}
    pos::Vector{SVector{D,F}}
end

function (pivstrat::FillDistance{D,F})(idcs::AbstractArray{Int}) where {D,F}
    return FillDistanceFunctor(zeros(F, length(idcs)), pivstrat.pos[idcs])
end

function (pivstrat::Union{Leja2Functor{D,F},FillDistanceFunctor{D,F}})() where {D,F}
    @views pivstrat.h .= norm.(pivstrat.pos .- Scalar(pivstrat.pos[1]))

    return 1
end

function (pivstrat::FillDistanceFunctor{D,F})(::AbstractArray) where {D,F}
    nextidx = argmax(pivstrat.h)
    maxval = pivstrat.h[nextidx]

    for k in eachindex(pivstrat.h)
        newfd = 0.0
        for (ind, pos) in enumerate(pivstrat.pos)
            if pivstrat.h[ind] > norm(pivstrat.pos[k] - pos)
                newfd < norm(pivstrat.pos[k] - pos) && (newfd = norm(pivstrat.pos[k] - pos))
            else
                newfd < pivstrat.h[ind] && (newfd = pivstrat.h[ind])
            end
        end
        newfd < maxval && (nextidx = k; maxval = newfd)
    end

    AdaptiveCrossApproximation.leja2!(pivstrat, nextidx)

    return nextidx
end
