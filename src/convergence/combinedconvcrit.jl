
mutable struct CombinedConvCrit
    crits::Vector{ConvCrit}
    isconverged::Vector{Bool}
end

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
