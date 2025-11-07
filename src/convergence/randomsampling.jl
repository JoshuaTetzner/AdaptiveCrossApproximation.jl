"""
    RandomSampling{F<:Real} <: ConvCrit

Convergence criterion based on random matrix entry sampling.
Monitors approximation error at randomly sampled positions.

# Fields

  - `nsamples::Int`: Number of random samples to take
  - `factor::F`: Factor for automatic sample count (nsamples = factor * (nrows + ncols))
  - `tol::F`: Convergence tolerance
"""
struct RandomSampling{F<:Real} <: ConvCrit
    nsamples::Int
    factor::F
    tol::F
end

"""
    RandomSamplingFunctor{F<:Real,K} <: ConvCritFunctor

Stateful random sampling convergence checker.
Tracks residual error at sampled matrix entries across iterations.

# Fields

  - `normUV²::F`: Squared Frobenius norm of approximation
  - `indices::Matrix{Int}`: Sampled matrix positions (nsamples × 2)
  - `rest::Vector{K}`: Residual values at sampled positions
  - `tol::F`: Convergence tolerance
"""
mutable struct RandomSamplingFunctor{F<:Real,K} <: ConvCritFunctor
    normUV²::F
    indices::Matrix{Int}
    rest::Vector{K}
    tol::F
end

"""
    RandomSampling(; factor=1.0, nsamples=0, tol=1e-4)

Construct random sampling convergence criterion.

# Arguments

  - `factor::F`: Multiplier for automatic sample count (default: `1.0`)
  - `nsamples::Int`: Fixed sample count (default: `0`, use factor instead)
  - `tol::F`: Convergence tolerance (default: `1e-4`)
"""
function RandomSampling(; factor::F=1.0, nsamples::Int=0, tol::F=1e-4) where {F<:Real}
    return RandomSampling(nsamples, factor, tol)
end

"""
    (cc::RandomSampling)(K::AbstractMatrix{T}, rowidcs, colidcs)

Initialize random sampling functor with sampled matrix entries.

# Arguments

  - `K::AbstractMatrix{T}`: Matrix to compress
  - `rowidcs::AbstractArray{Int}`: Active row indices
  - `colidcs::AbstractArray{Int}`: Active column indices
"""
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

"""
    tolerance(cc::RandomSamplingFunctor)

Get tolerance from random sampling functor.
"""
tolerance(cc::RandomSamplingFunctor) = cc.tol

"""
    (convcrit::RandomSamplingFunctor)(rowbuffer, colbuffer, npivot, maxrows, maxcolumns)

Check convergence using random sampling.
Updates residuals at sampled positions and compares to tolerance.

# Arguments

  - `rowbuffer::AbstractMatrix{K}`: Row factor buffer
  - `colbuffer::AbstractMatrix{K}`: Column factor buffer
  - `npivot::Int`: Current pivot index
  - `maxrows::Int`: Number of active rows
  - `maxcolumns::Int`: Number of active columns

# Returns

  - `npivot::Int`: Final pivot count
  - `continue::Bool`: Whether to continue iteration
"""
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
