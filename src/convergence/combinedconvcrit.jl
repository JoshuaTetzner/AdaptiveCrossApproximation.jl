"""
    CombinedConvCrit

Composite convergence criterion combining multiple criteria.
Converges when any constituent criterion is satisfied.

# Fields

  - `crits::Vector{ConvCrit}`: Vector of convergence criteria to combine
  - `isconverged::Vector{Bool}`: Convergence status for each criterion
"""
mutable struct CombinedConvCrit
    crits::Vector{ConvCrit}
    isconverged::Vector{Bool}
end

"""
    (convcrit::CombinedConvCrit)(rowbuffer, colbuffer, npivot, maxrows, maxcolumns)

Check convergence using all combined criteria.
Returns when any criterion signals convergence.

# Arguments

  - `rowbuffer::AbstractMatrix{K}`: Row factor buffer
  - `colbuffer::AbstractMatrix{K}`: Column factor buffer
  - `npivot::Int`: Current pivot index
  - `maxrows::Int`: Number of active rows
  - `maxcolumns::Int`: Number of active columns

# Returns

  - `npivot::Int`: Final pivot count
  - `continue::Bool`: Whether to continue iteration (true if any criterion satisfied)
"""
function (convcrit::CombinedConvCrit)(
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
        npivot_ != npivot && return npivot_, convcrit.isconverged[i]
    end
    return npivot, any(convcrit.isconverged)
end

"""
    (convcrit::CombinedConvCrit)(K::AbstractMatrix, rowidcs, colidcs)

Initialize combined criterion functors.
Handles special initialization for sampling-based criteria.

# Arguments

  - `K::AbstractMatrix`: Matrix to compress
  - `rowidcs::AbstractArray{Int}`: Active row indices
  - `colidcs::AbstractArray{Int}`: Active column indices
"""
function (convcrit::CombinedConvCrit)(
    K::AbstractMatrix, rowidcs::AbstractArray{Int}, colidcs::AbstractArray{Int}
)
    curr_crits = Vector{ConvCrit}(undef, length(convcrit.crits))
    for (i, crit) in enumerate(convcrit.crits)
        if isa(crit, RandomSampling)
            curr_crits[i] = crit(K, rowidcs, colidcs)
        else
            curr_crits[i] = crit()
        end
    end
    return CombinedConvCrit(curr_crits, zeros(Bool, length(curr_crits)))
end
