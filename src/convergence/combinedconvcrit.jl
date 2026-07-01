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

mutable struct CombinedConvCritFunctor <: ConvCritFunctor
    crits::Vector{ConvCritFunctor}
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
        curr_crits[i] = _buildconvcrit(crit, K, nrowidcs, ncolidcs, maxrank)
    end
    return CombinedConvCritFunctor(curr_crits, ones(Bool, length(curr_crits)))
end

function (convcrit::CombinedConvCrit)(
    K::Union{AbstractMatrix,AbstractKernelMatrix},
    rowidcs::AbstractVector{Int},
    colidcs::AbstractVector{Int};
    maxrank::Int=40,
)
    curr_crits = Vector{ConvCritFunctor}(undef, length(convcrit.crits))
    for (i, crit) in enumerate(convcrit.crits)
        curr_crits[i] = _buildconvcrit(crit, K, rowidcs, colidcs, maxrank)
    end
    return CombinedConvCritFunctor(curr_crits, ones(Bool, length(curr_crits)))
end

_buildconvcrit(cc::CombinedConvCrit, A, rowidcs, colidcs, maxrank) =
    cc(A, rowidcs, colidcs; maxrank=maxrank)

_buildconvcrit(
    cc::CombinedConvCrit,
    A,
    rowidcs::AbstractVector{Int},
    colidcs::AbstractVector{Int},
    maxrank,
) = cc(A, rowidcs, colidcs; maxrank=maxrank)

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
