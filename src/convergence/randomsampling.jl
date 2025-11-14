struct RandomSampling{F<:Real} <: ConvCrit
    nsamples::Int
    factor::F
    tol::F
end

mutable struct RandomSamplingFunctor{F<:Real,K} <: ConvCritFunctor
    normUV²::F
    indices::Matrix{Int}
    rest::Vector{K}
    tol::F
end

function RandomSampling(; factor::F=1.0, nsamples::Int=0, tol::F=1e-4) where {F<:Real}
    return RandomSampling(nsamples, factor, tol)
end

function (cc::RandomSampling)(
    K::AbstractMatrix{T}, rowidcs::AbstractArray{Int}, colidcs::AbstractArray{Int}
) where {T}
    rowlen = length(rowidcs)
    collen = length(colidcs)
    nsamples = cc.nsamples == 0 ? cc.factor * (rowlen + collen) : cc.nsamples
    indices = hcat(rand(1:rowlen, nsamples), rand(1:collen, nsamples))
    rest = [K[rc[1], rc[2]][1] for rc in eachrow(indices)]
    return RandomSamplingFunctor(0.0, indices, rest, cc.tol)
end

tolerance(cc::RandomSamplingFunctor) = cc.tol

function (convcrit::RandomSamplingFunctor{F,K})(
    rowbuffer::AbstractMatrix{K},
    colbuffer::AbstractMatrix{K},
    npivot::Int,
    maxrows::Int,
    maxcolumns::Int,
) where {F<:Real,K}

    # omit this to increase performance (safty measures)---
    @views rnorm = norm(rowbuffer[npivot, 1:maxcolumns])
    @views cnorm = norm(colbuffer[1:maxrows, npivot])

    (isapprox(rnorm, 0.0) && isapprox(cnorm, 0.0)) && (return npivot - 1, false)
    ((isapprox(rnorm, 0.0) || isapprox(cnorm, 0.0)) && !(npivot == 1)) &&
        (return npivot - 1, false)
    # -----------------------------------------------------
    for i in eachindex(convcrit.rest)
        @views convcrit.rest[i] -=
            colbuffer[convcrit.indices[i, 1], npivot] *
            rowbuffer[npivot, convcrit.indices[i, 2]]
    end
    meanrest = sum(abs.(convcrit.rest) .^ 2) / convcrit.nsamples

    normF!(convcrit, rowbuffer, colbuffer, npivot, maxrows, maxcolumns)
    return npivot,
    sqrt(meanrest * maxrows * maxcolumns) > convcrit.tol * sqrt(convcrit.normUV²)
end
