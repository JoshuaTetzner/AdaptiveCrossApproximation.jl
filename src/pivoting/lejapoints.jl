
mutable struct Leja2{D,F} <: GeoPivStrat
    h::Vector{F}
    pos::Vector{SVector{D,F}}
    usedidcs::Vector{Bool}
end


function leja2!(fd::Leja2, nextidx::Int)
    newh = norm.(fd.pos .- Scalar(fd.pos[nextidx]))
    for idx in eachindex(fd.h)
        fd.h[idx] > newh[idx] && (fd.h[idx] = newh[idx])
    end
end

function Leja2!(pos::Vector{SVector{D,F}}) where {D,F}
    ModifiedFillDistance(F[], pos, Bool[])
end

function (pivstrat::Leja2{D,F})(idcs::Vector{Int}) where {D,F}
    return Leja2(
        zeros(F, length(rcp)), pivstrat.pos[rcp], zeros(Bool, length(rcp))
    )
end

function (pivstrat::Leja2{D,F})(::AbstractArray) where {D,F}
    nextidx = argmax(h)
    filldistance!(pivstrat, nextidx)
    pivstrat.usedidcs[nextidx] = true

    return nextidx
end
