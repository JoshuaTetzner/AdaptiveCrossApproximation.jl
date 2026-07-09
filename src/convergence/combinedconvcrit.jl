"""
    CombinedConvCrit

Composite convergence criterion combining multiple criteria.
Converges when any constituent criterion is satisfied.

# Fields

  - `crits::Vector{ConvCrit}`: Vector of convergence criteria to combine
"""
mutable struct CombinedConvCrit <: ConvCrit
    crits::Vector{ConvCrit}
end

# `crits` is a Tuple (not Vector) of the *concrete* per-criterion functors, so
# iterating it and calling each criterion is type-stable: the per-pivot
# convergence call returns a concrete `Tuple{Int,Bool}`. With a
# `Vector{ConvCritFunctor}` (abstract eltype) that return widens to
# `Tuple{Any,Bool}`, which makes `npivot` ::Any in the ACA main loop and boxes
# every deflation inner-loop iteration - billions of allocations on large
# far-field assemblies. The construction below (once per task) materializes the
# tuple; the hot path is then fully concrete.
mutable struct CombinedConvCritFunctor{T<:Tuple} <: ConvCritFunctor
    crits::T
    isconverged::Vector{Bool}
end

function (convcrit::CombinedConvCrit)(
    K::Union{AbstractMatrix,AbstractKernelMatrix},
    nrowidcs::Int,
    ncolidcs::Int;
    maxrank::Int=40,
)
    curr_crits = Vector{ConvCritFunctor}(undef, length(convcrit.crits))
    for (i, crit) in enumerate(convcrit.crits)
        if isa(crit, RandomSampling)
            curr_crits[i] = crit(K, nrowidcs, ncolidcs)
        elseif isa(crit, FNormExtrapolatorFunctor)
            curr_crits[i] = crit(maxrank)
        else
            curr_crits[i] = crit()
        end
    end
    return CombinedConvCritFunctor(tuple(curr_crits...), ones(Bool, length(curr_crits)))
end

_buildconvcrit(cc::CombinedConvCrit, A, rowidcs, colidcs, maxrank) =
    cc(A, rowidcs, colidcs; maxrank=maxrank)

_buildconvcrit(
    cc::CombinedConvCrit,
    A,
    rowidcs::AbstractVector{Int},
    colidcs::AbstractVector{Int},
    maxrank,
) = cc(A, length(rowidcs), length(colidcs); maxrank=maxrank)

function reset!(
    convcrit::CombinedConvCritFunctor,
    rowidcs::AbstractVector{Int},
    colidcs::AbstractVector{Int},
)
    for crit in convcrit.crits
        reset!(crit, rowidcs, colidcs)
    end
    fill!(convcrit.isconverged, true)
    return convcrit
end

function (convcrit::CombinedConvCritFunctor)(
    rowbuffer::AbstractMatrix{K},
    colbuffer::AbstractMatrix{K},
    npivot::Int,
    maxrows::Int,
    maxcolumns::Int,
) where {K}
    for (i, crit) in enumerate(convcrit.crits)
        npivot_, convcrit.isconverged[i] = crit(
            rowbuffer, colbuffer, npivot, maxrows, maxcolumns
        )

        if (npivot_ != npivot && i == length(convcrit.crits))
            rowbuffer[npivot, :] .= K(0)
            colbuffer[:, npivot] .= K(0)
            return npivot_, convcrit.isconverged[i]
        end
    end

    return npivot, any(convcrit.isconverged)
end
