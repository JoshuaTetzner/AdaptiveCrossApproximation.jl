# RandomSampling convergence does need an inizializer

mutable struct RandomSampling{F<:Real,K} <: ConvCrit
    normUV²::F
    nsamples::Int
    factor::F
    indices::Matrix{Int}
    rest::Vector{K}
    tol::F
end

function RandomSampling(
    ::Type{K}; factor::F=1.0, nsamples::Int=0, tol::F=1e-4
) where {K,F<:Real}
    return RandomSampling{F,K}(F(0.0), nsamples, factor, zeros(Int, 0, 0), zeros(K, 0), tol)
end

tolerance(cc::RandomSampling) = cc.tol

function (convcrit::RandomSampling{F,K})(
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
