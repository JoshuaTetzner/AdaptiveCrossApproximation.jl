
struct Leja2{D,F<:Real} <: GeoPivStrat
    h::Vector{F}
    pos::Vector{SVector{D,F}}
end

function leja2!(pivstrat::GeoPivStrat, nextidx::Int)
    newh = norm.(pivstrat.pos .- Scalar(pivstrat.pos[nextidx]))
    all(==(0.0), pivstrat.h) && (pivstrat.h .= newh)
    for idx in eachindex(pivstrat.h)
        pivstrat.h[idx] > newh[idx] && (pivstrat.h[idx] = newh[idx])
    end
end

function Leja2(pos::Vector{SVector{D,F}}) where {D,F}
    return Leja2{D,F}(F[], pos)
end

function (pivstrat::Leja2{D,F})(idcs::Vector{Int}) where {D,F}
    return Leja2{D,F}(zeros(F, length(idcs)), pivstrat.pos[idcs])
end

function (pivstrat::Leja2{D,F})(::AbstractArray) where {D,F}
    nextidx = argmax(pivstrat.h)
    leja2!(pivstrat, nextidx)

    return nextidx
end
