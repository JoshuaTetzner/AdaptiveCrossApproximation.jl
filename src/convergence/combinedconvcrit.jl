"""
    CombinedConvCrit

Composite convergence criterion combining multiple criteria.
Stops only when ALL constituent criteria have signaled convergence (AND logic).

# Fields

  - `crits::Vector{ConvCrit}`: Vector of convergence criteria to combine
"""
mutable struct CombinedConvCrit <: ConvCrit
    crits::Vector{ConvCrit}
end

struct CombinedConvCritFunctor <: ConvCritFunctor
    crits::Vector{ConvCritFunctor}
    isconverged::Vector{Bool}
end

function _buildconvcrit(cc::CombinedConvCrit, K, rowidcs, colidcs, maxrank)
    curr_crits = Vector{ConvCritFunctor}(undef, length(cc.crits))
    for (i, crit) in enumerate(cc.crits)
        curr_crits[i] = _buildconvcrit(crit, K, rowidcs, colidcs, maxrank)
    end
    return CombinedConvCritFunctor(curr_crits, ones(Bool, length(curr_crits)))
end

function (convcrit::CombinedConvCrit)(
    K::Union{AbstractMatrix,AbstractKernelMatrix},
    rowidcs::AbstractVector{Int},
    colidcs::AbstractVector{Int};
    maxrank::Int=40,
)
    return _buildconvcrit(convcrit, K, rowidcs, colidcs, maxrank)
end

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
        if npivot_ != npivot && i == length(convcrit.crits)
            rowbuffer[npivot, :] .= K(0)
            colbuffer[:, npivot] .= K(0)
            return npivot_, convcrit.isconverged[i]
        end
    end
    return npivot, any(convcrit.isconverged)
end
