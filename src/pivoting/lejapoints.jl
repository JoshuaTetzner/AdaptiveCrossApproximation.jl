
struct Leja2{D,F<:Real} <: GeoPivStrat
    pos::Vector{SVector{D,F}}
end

struct Leja2Functor{D,F<:Real} <: PivStratFunctor
    h::Vector{F}
    pos::Vector{SVector{D,F}}
end

function (pivstrat::Leja2{D,F})(idcs::AbstractArray{Int}) where {D,F}
    return Leja2Functor{D,F}(zeros(F, length(idcs)), pivstrat.pos[idcs])
end

function leja2!(pivstrat::GeoPivStratFunctor, nextidx::Int)
    newh = norm.(pivstrat.pos .- Scalar(pivstrat.pos[nextidx]))
    all(==(0.0), pivstrat.h) && (pivstrat.h .= newh)
    for idx in eachindex(pivstrat.h)
        pivstrat.h[idx] > newh[idx] && (pivstrat.h[idx] = newh[idx])
    end
end

function (pivstrat::Leja2Functor{D,F})(::AbstractArray) where {D,F}
    nextidx = argmax(pivstrat.h)
    leja2!(pivstrat, nextidx)

    return nextidx
end
