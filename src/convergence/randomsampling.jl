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
    RandomSamplingFunctor{F<:Real,K,M} <: ConvCritFunctor

Stateful random sampling convergence checker.
Tracks residual error at sampled matrix entries across iterations.

# Fields

    - `normUV²::F`: Squared Frobenius norm of approximation
    - `indices::Vector{Tuple{Int,Int}}`: Sampled matrix positions
    - `rest::Vector{K}`: Residual values at sampled positions
    - `tol::F`: Convergence tolerance
    - `mat::M`: Matrix handle used for refreshing random samples
"""
mutable struct RandomSamplingFunctor{F<:Real,K,M} <: ConvCritFunctor
    normUV²::F
    nsamples::Int
    factor::F
    nactive::Int
    indices::Vector{Tuple{Int,Int}}
    rest::Vector{K}
    tol::F
    mat::M
end

@inline function _samplecount(cc::RandomSampling, rowlen::Int, collen::Int)
    nsamples = cc.nsamples == 0 ? Int(round(cc.factor * (rowlen + collen))) : cc.nsamples
    nsamples = max(1, nsamples)
    return min(nsamples, rowlen * collen)
end

@inline function _samplecount(convcrit::RandomSamplingFunctor, rowlen::Int, collen::Int)
    nsamples = if convcrit.nsamples == 0
        Int(round(convcrit.factor * (rowlen + collen)))
    else
        convcrit.nsamples
    end
    nsamples = max(1, nsamples)
    return min(nsamples, rowlen * collen)
end

function _sample_indices(rowlen::Int, collen::Int, nsamples::Int)
    idxset = Set{Tuple{Int,Int}}()
    sizehint!(idxset, nsamples)
    while length(idxset) < nsamples
        push!(idxset, (rand(1:rowlen), rand(1:collen)))
    end
    return collect(idxset)
end

function _sample_indices!(
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

@inline _entryvalue(entry::Number) = entry
@inline _entryvalue(entry) = entry[1]

function _fill_rest!(
    rest::AbstractVector,
    indices::Vector{Tuple{Int,Int}},
    K::AbstractMatrix,
    rowidcs::AbstractArray{Int},
    colidcs::AbstractArray{Int},
    nactive::Int,
)
    @inbounds for i in 1:nactive
        rc = indices[i]
        rest[i] = _entryvalue(K[rowidcs[rc[1]], colidcs[rc[2]]])
    end
    return rest
end

function _restbuffer(
    K::Union{AbstractMatrix,AbstractKernelMatrix},
    rowidcs::AbstractArray{Int},
    colidcs::AbstractArray{Int},
    indices::Vector{Tuple{Int,Int}},
)
    return Vector{eltype(K)}(undef, length(indices))
end

function _fill_rest!(
    rest::AbstractVector,
    indices::Vector{Tuple{Int,Int}},
    K::AbstractKernelMatrix,
    rowidcs::AbstractArray{Int},
    colidcs::AbstractArray{Int},
    nactive::Int,
)
    @inbounds for i in 1:nactive
        rc = indices[i]
        @views K(rest[i:i], rowidcs[rc[1]:rc[1]], colidcs[rc[2]:rc[2]])
    end
    return rest
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

function (cc::RandomSampling)(
    K::Union{AbstractMatrix,AbstractKernelMatrix},
    rowidcs::AbstractArray{Int},
    colidcs::AbstractArray{Int},
)
    rowlen = length(rowidcs)
    collen = length(colidcs)
    nsamples = _samplecount(cc, rowlen, collen)
    indices = _sample_indices(rowlen, collen, nsamples)
    rest = _restbuffer(K, rowidcs, colidcs, indices)
    _fill_rest!(rest, indices, K, rowidcs, colidcs, nsamples)
    return RandomSamplingFunctor(
        zero(cc.tol), cc.nsamples, cc.factor, nsamples, indices, rest, cc.tol, K
    )
end

function reset!(convcrit::RandomSamplingFunctor)
    convcrit.normUV² = zero(convcrit.normUV²)
    fill!(view(convcrit.rest, 1:(convcrit.nactive)), zero(eltype(convcrit.rest)))
    fill!(view(convcrit.indices, 1:(convcrit.nactive)), (0, 0))
    return nothing
end

function reset!(
    convcrit::RandomSamplingFunctor,
    rowidcs::AbstractArray{Int},
    colidcs::AbstractArray{Int},
)
    resize!(convcrit, rowidcs, colidcs)
    return nothing
end

function Base.resize!(
    convcrit::RandomSamplingFunctor,
    rowidcs::AbstractArray{Int},
    colidcs::AbstractArray{Int},
)
    rowlen = length(rowidcs)
    collen = length(colidcs)
    nsamples = _samplecount(convcrit, rowlen, collen)

    _sample_indices!(convcrit.indices, rowlen, collen, nsamples)
    length(convcrit.rest) < nsamples && resize!(convcrit.rest, nsamples)
    _fill_rest!(convcrit.rest, convcrit.indices, convcrit.mat, rowidcs, colidcs, nsamples)

    convcrit.nactive = nsamples
    convcrit.normUV² = zero(convcrit.normUV²)
    return convcrit
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
function (convcrit::RandomSamplingFunctor{F,K,M})(
    rowbuffer::AbstractMatrix{K},
    colbuffer::AbstractMatrix{K},
    npivot::Int,
    maxrows::Int,
    maxcolumns::Int,
) where {F<:Real,K,M}

    # omit this to increase performance (safty measures)---
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
