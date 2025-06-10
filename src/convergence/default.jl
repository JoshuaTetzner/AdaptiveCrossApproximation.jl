function normF(
    normUV²::F,
    rowbuffer::AbstractMatrix{K},
    colbuffer::AbstractMatrix{K},
    npivot::Int,
    maxrows::Int,
    maxcolumns::Int
) where {F<:Real,K}
    @views normUV² += (
        norm(rowbuffer[npivot, 1:maxcolumns]) * norm(colbuffer[1:maxrows, npivot])
    )^2

    for j = 1:(npivot-1)
        @views normUV² += 2 * real.(
            dot(colbuffer[1:maxrows, npivot], colbuffer[1:maxrows, j]
            ) * dot(rowbuffer[npivot, 1:maxcolumns], rowbuffer[j, 1:maxcolumns]))
    end

    return normUV²
end

#Default

mutable struct Default{F} <: ConvCrit
    normUV²::F
end

(::Default{F})(kwargs...) where {F} = Default(0.0)

function (convcrit::Default{F})(
    rowbuffer::AbstractMatrix{K},
    colbuffer::AbstractMatrix{K},
    npivot::Int,
    maxrows::Int,
    maxcolumns::Int,
    tol::F
) where {F<:Real,K}

    rnorm = norm(rowbuffer[npivot, 1:maxcolumns])
    cnorm = norm(colbuffer[1:maxrows, npivot])

    @views if isapprox(rnorm, 0.0) || isapprox(cnorm, 0.0)
        if isapprox(rnorm, 0.0) && isapprox(cnorm, 0.0)
            return npivot - 1, false
        else
            return npivot - 1, true
        end
    else

        convcrit.normUV² = normF(
            convcrit.normUV², rowbuffer, colbuffer, npivot, maxrows, maxcolumns
        )
        return npivot, rnorm * cnorm > tol * sqrt(convcrit.normUV²)
    end
end
