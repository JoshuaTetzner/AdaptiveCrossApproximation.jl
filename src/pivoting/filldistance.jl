
struct FillDistance{D,F<:Real} <: GeoPivStrat
    h::Vector{F}
    pos::Vector{SVector{D,F}}
end

function FillDistance(pos::Vector{SVector{D,F}}) where {D,F<:Real}
    return FillDistance{D,F}(F[], pos)
end

function (pivstrat::FillDistance{D,F})(idcs::Vector{Int}) where {D,F}
    return FillDistance{D,F}(zeros(F, length(idcs)), pivstrat.pos[idcs])
end

function (pivstrat::Union{Leja2{D,F},FillDistance{D,F}})() where {D,F}
    @views pivstrat.h .= norm.(pivstrat.pos .- Scalar(pivstrat.pos[1]))

    return 1
end

function (pivstrat::FillDistance{D,F})(::AbstractArray) where {D,F}
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
        newfd < maxval && (nextidx=k; maxval=newfd)
    end

    AdaptiveCrossApproximation.leja2!(pivstrat, nextidx)

    return nextidx
end
