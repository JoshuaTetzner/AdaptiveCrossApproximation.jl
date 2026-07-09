"""
    ConvCrit

Abstract base type for convergence criteria used by ACA and iACA compressors.

# Notes

Concrete subtypes define how stopping decisions are made and are converted into
stateful `ConvCritFunctor` objects during block compression.

# See also

`ConvCritFunctor`, `FNormEstimator`, `iFNormEstimator`, `FNormExtrapolator`, `RandomSampling`
"""
abstract type ConvCrit end

"""
    ConvCritFunctor

Abstract base type for stateful convergence criterion functors.

# Notes

Instances are called during ACA iterations and return `(npivot, continue::Bool)`.
Subtypes should implement `reset!` to reinitialize internal state for a new block.

# See also

`ConvCrit`, `reset!`, `normF!`
"""
abstract type ConvCritFunctor end

function (cc::ConvCrit)(
    K::Union{AbstractMatrix,AbstractKernelMatrix},
    rowidcs::AbstractArray{Int},
    colidcs::AbstractArray{Int};
    maxrank::Int=40,
)
    if isa(cc, CombinedConvCrit)
        return cc(K, rowidcs, colidcs; maxrank=maxrank)
    elseif isa(cc, RandomSampling)
        return cc(K, rowidcs, colidcs)
    elseif isa(cc, FNormExtrapolator)
        return cc(maxrank)
    else
        return cc()
    end
end

"""
    reset!(convcrit::ConvCritFunctor)

Reset a convergence functor before starting compression of a new block.

# Notes

Concrete subtypes should overload this method. The default fallback throws
`ArgumentError`.

# See also

`ConvCritFunctor`
"""
function reset!(convcrit::ConvCritFunctor)
    return throw(ArgumentError("reset! is not implemented for $(typeof(convcrit))."))
end

function reset!(convcrit::ConvCritFunctor, args...)
    return reset!(convcrit)
end

# Allocation-free reductions over the active part of an ACA buffer. They match
# norm()/dot() but index the matrix directly instead of materializing a
# SubArray, which - inside the per-pivot convergence check - otherwise escapes
# into norm()/dot() and heap-allocates once per pivot (the dominant far-field
# allocation at scale). The arithmetic is identical: colnorm/rownorm compute
# sqrt(sum(abs2)) and coldot/rowdot the Hermitian dot sum(conj(x)*y).
@inline function colnorm(A::AbstractMatrix{K}, col::Int, m::Int) where {K}
    s = zero(real(K))
    @inbounds @simd for i in 1:m
        s += abs2(A[i, col])
    end
    return sqrt(s)
end
@inline function rownorm(A::AbstractMatrix{K}, row::Int, n::Int) where {K}
    s = zero(real(K))
    @inbounds for j in 1:n
        s += abs2(A[row, j])
    end
    return sqrt(s)
end
@inline function coldot(A::AbstractMatrix{K}, p::Int, q::Int, m::Int) where {K}
    s = zero(K)
    @inbounds @simd for i in 1:m
        s += conj(A[i, p]) * A[i, q]
    end
    return s
end
@inline function rowdot(A::AbstractMatrix{K}, p::Int, q::Int, n::Int) where {K}
    s = zero(K)
    @inbounds for j in 1:n
        s += conj(A[p, j]) * A[q, j]
    end
    return s
end
# Vector counterpart of colnorm/rownorm: sqrt(sum(abs2)) over a row/column buffer
# slice without materializing it. The iACA paths previously passed
# `buffer[npivot, 1:n]` (a heap copy) into `norm`; with a `view` the slice no
# longer copies, and this helper keeps the reduction allocation-free and SIMD'd.
@inline function vecnorm(v::AbstractVector{K}) where {K}
    s = zero(real(K))
    @inbounds @simd for i in eachindex(v)
        s += abs2(v[i])
    end
    return sqrt(s)
end

function normF!(
    convcrit::ConvCritFunctor,
    rowbuffer::AbstractMatrix{K},
    colbuffer::AbstractMatrix{K},
    npivot::Int,
    maxrows::Int,
    maxcolumns::Int,
) where {K}
    rnorm = rownorm(rowbuffer, npivot, maxcolumns)
    cnorm = colnorm(colbuffer, npivot, maxrows)
    return normF!(convcrit, rowbuffer, colbuffer, npivot, maxrows, maxcolumns, rnorm, cnorm)
end

function normF!(
    convcrit::ConvCritFunctor,
    rowbuffer::AbstractMatrix{K},
    colbuffer::AbstractMatrix{K},
    npivot::Int,
    maxrows::Int,
    maxcolumns::Int,
    rnorm,
    cnorm,
) where {K}
    convcrit.normUV² += (rnorm * cnorm)^2

    for j in 1:(npivot - 1)
        convcrit.normUV² +=
            2 * real(
                coldot(colbuffer, npivot, j, maxrows) *
                rowdot(rowbuffer, npivot, j, maxcolumns),
            )
    end
end

function normF!(
    convcrit::ConvCritFunctor, rcbuffer::AbstractVector{K}, npivot::Int
) where {K}
    return convcrit.normUV = ((npivot - 1) * convcrit.normUV + vecnorm(rcbuffer)) / npivot
end
