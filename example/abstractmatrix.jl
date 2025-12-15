
using LinearAlgebra

struct OneOverRKernel{F} <: AbstractMatrix{F}
    rowpoints::Vector{SVector{3,F}}
    colvectors::Vector{SVector{3,F}}
end

function Base.size(A::OneOverRKernel, dim=nothing)
    if dim === nothing
        return (length(A.rowpoints), length(A.colvectors))
    elseif dim == 1
        return length(A.rowpoints)
    elseif dim == 2
        return length(A.colvectors)
    else
        error("dim must be either 1 or 2")
    end
end

function Base.getindex(A::OneOverRKernel{F}, i::Int, j::Int) where {F}
    return A(i, j)
end

function (m::OneOverRKernel{F})(i::Int, j::Int) where {F}
    r = norm(m.rowpoints[i] - m.colvectors[j])
    return 1.0 / r
end

function (m::OneOverRKernel{F})(
    buf::Matrix{F}, i::AbstractVector{Int}, j::AbstractVector{Int}
) where {F}
    for ii in i
        for jj in j
            r = norm(m.rowpoints[ii] - m.colvectors[jj])
            buf[ii, jj] += 1.0 / r
        end
    end

    return buf
end

using AdaptiveCrossApproximation

pts = rand(SVector{3,Float64}, 100)
pts2 = rand(SVector{3,Float64}, 100)
A = OneOverRKernel{Float64}(pts, pts2)

U, V = AdaptiveCrossApproximation.aca(A; maxrank=100)
using LinearAlgebra
norm(A - U * V) / norm(A)
