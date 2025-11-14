abstract type ConvCrit end
abstract type ConvCritFunctor end

# ACA
function normF!(
    convcrit::ConvCritFunctor,
    rowbuffer::AbstractMatrix{K},
    colbuffer::AbstractMatrix{K},
    npivot::Int,
    maxrows::Int,
    maxcolumns::Int,
) where {K}
    @views convcrit.normUV² +=
        (norm(rowbuffer[npivot, 1:maxcolumns]) * norm(colbuffer[1:maxrows, npivot]))^2

    for j in 1:(npivot - 1)
        @views convcrit.normUV² +=
            2 *
            real.(
                dot(colbuffer[1:maxrows, npivot], colbuffer[1:maxrows, j]) *
                dot(rowbuffer[npivot, 1:maxcolumns], rowbuffer[j, 1:maxcolumns]),
            )
    end
end

# iACA
function normF!(
    convcrit::ConvCritFunctor, rcbuffer::AbstractVector{K}, npivot::Int
) where {K}
    return convcrit.normUV = ((npivot - 1) * convcrit.normUV + norm(rcbuffer)) / npivot
end
