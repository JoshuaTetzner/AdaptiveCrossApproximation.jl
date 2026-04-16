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

mutable struct RandomSamplingFunctor{F<:Real,K,M,T} <: ConvCritFunctor
    convergence::T
    mat::M
    nactive::Int
    normUV²::F
    indices::Vector{Tuple{Int,Int}}
    rest::Vector{K}
end

tolerance(cc::RandomSamplingFunctor) = cc.convergence.tol

@inline function _samplecount(cc::RandomSampling, rowlen::Int, collen::Int)
    nsamples = cc.nsamples == 0 ? Int(round(cc.factor * (rowlen + collen))) : cc.nsamples
    nsamples = max(1, nsamples)
    return min(nsamples, rowlen * collen)
end

function _sampleindices(rowlen::Int, collen::Int, nsamples::Int)
    idxset = Vector{Tuple{Int,Int}}(undef, nsamples)
    for i in 1:nsamples
        idxset[i] = (rand(1:rowlen), rand(1:collen))
    end
    return idxset
end

function _sampleindices!(
    indices::Vector{Tuple{Int,Int}}, rowlen::Int, collen::Int, nsamples::Int
)
    length(indices) < nsamples && resize!(indices, nsamples)
    @inbounds for i in 1:nsamples
        while true
            rc = (rand(1:rowlen), rand(1:collen))
            duplicate = false
            for j in 1:(i - 1)
                if indices[j] == rc
                    duplicate = true
                    break
                end
            end
            duplicate && continue
            indices[i] = rc
            break
        end
    end
    return indices
end

function _fillrest!(
    rest::AbstractVector,
    indices::Vector{Tuple{Int,Int}},
    K::AbstractMatrix,
    rowidcs::AbstractArray{Int},
    colidcs::AbstractArray{Int},
    nactive::Int,
)
    length(rest) < nactive && resize!(rest, nactive)
    @inbounds for i in 1:nactive
        rc = indices[i]
        rest[i] = K[rowidcs[rc[1]], colidcs[rc[2]]]
    end
end

function _fillrest!(
    rest::AbstractVector,
    indices::Vector{Tuple{Int,Int}},
    K::AbstractKernelMatrix,
    rowidcs::AbstractArray{Int},
    colidcs::AbstractArray{Int},
    nactive::Int,
)
    length(rest) < nactive && resize!(rest, nactive)
    @inbounds for i in 1:nactive
        rc = indices[i]
        @views K(rest[i:i], rowidcs[rc[1]:rc[1]], colidcs[rc[2]:rc[2]])
    end
end

function (cc::RandomSampling)(
    K::Union{AbstractMatrix,AbstractKernelMatrix},
    rowidcs::AbstractArray{Int},
    colidcs::AbstractArray{Int},
)
    rowlen = length(rowidcs)
    collen = length(colidcs)
    nsamples = _samplecount(cc, rowlen, collen)
    indices = _sampleindices(rowlen, collen, nsamples)
    rest = zeros(eltype(K), nsamples)
    _fillrest!(rest, indices, K, rowidcs, colidcs, nsamples)
    return RandomSamplingFunctor(cc, K, nsamples, zero(cc.tol), indices, rest)
end

function (cc::RandomSampling)(
    K::Union{AbstractMatrix,AbstractKernelMatrix}, rowlen::Int, collen::Int
)
    nsamples = _samplecount(cc, rowlen, collen)
    indices = Vector{Tuple{Int,Int}}(undef, nsamples)
    rest = zeros(eltype(K), nsamples)
    return RandomSamplingFunctor(cc, K, nsamples, zero(cc.tol), indices, rest)
end

_buildconvcrit(cc::RandomSampling, K, rowidcs, colidcs, maxrank) = cc(K, rowidcs, colidcs)

function reset!(
    convcrit::RandomSamplingFunctor,
    rowidcs::AbstractArray{Int},
    colidcs::AbstractArray{Int},
)
    rowlen = length(rowidcs)
    collen = length(colidcs)
    nsamples = _samplecount(convcrit.convergence, rowlen, collen)
    _sampleindices!(convcrit.indices, rowlen, collen, nsamples)
    _fillrest!(convcrit.rest, convcrit.indices, convcrit.mat, rowidcs, colidcs, nsamples)
    convcrit.nactive = nsamples
    convcrit.normUV² = zero(convcrit.normUV²)
    return nothing
end

function (convcrit::RandomSamplingFunctor{F,K,M})(
    rowbuffer::AbstractMatrix{K},
    colbuffer::AbstractMatrix{K},
    npivot::Int,
    maxrows::Int,
    maxcolumns::Int,
) where {F<:Real,K,M}

    @views rnorm = norm(rowbuffer[npivot, 1:maxcolumns])
    @views cnorm = norm(colbuffer[1:maxrows, npivot])
    nactive = convcrit.nactive

    sumrest2 = zero(real(K))
    @inbounds for i in 1:nactive
        rc = convcrit.indices[i]
        convcrit.rest[i] -= colbuffer[rc[1], npivot] * rowbuffer[npivot, rc[2]]
        sumrest2 += abs2(convcrit.rest[i])
    end
    meanrest = sumrest2 / nactive

    (meanrest == 0.0 && rnorm == 0.0 && cnorm == 0.0) && (return npivot - 1, false)

    lhs = sqrt(meanrest * maxrows * maxcolumns)
    rhs = tolerance(convcrit) * sqrt(convcrit.normUV²)
    (rnorm == 0.0 || cnorm == 0.0) && (return npivot - 1, lhs > rhs)

    normF!(convcrit, rowbuffer, colbuffer, npivot, maxrows, maxcolumns)
    rhs = tolerance(convcrit) * sqrt(convcrit.normUV²)
    return npivot, lhs > rhs
end
