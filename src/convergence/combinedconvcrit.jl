
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
    tol::F,
) where {F<:Real,K}
    for (i, crit) in enumerate(convcrit.crits)
        npivot_, convcrit.isconverged[i] = crit(
            rowbuffer, colbuffer, npivot, maxrows, maxcolumns, tol
        )
        npivot_ != npivot && return npivot_, convcrit.isconverged[i]
    end
    return npivot, any(convcrit.isconverged)
end
