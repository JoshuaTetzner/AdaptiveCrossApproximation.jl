
struct ACA{RowPivType,ColPivType,ConvCritType}
    rowpivoting::RowPivType
    columnpivoting::ColPivType
    convergence::ConvCritType

    function ACA(rowpivoting, columnpivoting, convergence)
        return new{typeof(rowpivoting),typeof(columnpivoting),typeof(convergence)}(
            rowpivoting, columnpivoting, convergence
        )
    end
end

nextrc!(buf, A::AbstractArray, i, j) = (buf .= view(A, i, j))

function ACA(;
    rowpivoting=MaximumValue(), columnpivoting=MaximumValue(), convergence=Default(0.0)
)
    return ACA(rowpivoting, columnpivoting, convergence)
end

function (aca::ACA)(
    A,
    rowbuffer::AbstractMatrix{K},
    colbuffer::AbstractMatrix{K},
    maxrank::Int,
    tol::F,
    rowidcs=Vector(1:size(rowbuffer, 1)),
    colidcs=Vector(1:size(colbuffer, 2)),
) where {F<:Real,K}
    rows = Int[1]
    cols = Int[]
    maxrows, maxcolumns = size(M)
    npivot = 1
    nextrow = 1
    aca.rowpivoting.usedidcs[nextrow] = true

    nextrc!(
        view(rowbuffer, npivot:npivot, 1:maxcolumns), A, rowidcs[1:1], colidcs[1:maxcolumns]
    )
    nextcolumn = aca.columnpivoting(rowbuffer[npivot, 1:maxcolumns])
    push!(cols, nextcolumn)
    denominator = rowbuffer[npivot, nextcolumn]
    if denominator != 0.0
        @. rowbuffer[npivot, 1:maxcolumns] /= denominator
        nextrc!(
            view(colbuffer, 1:maxrows, npivot:npivot),
            A,
            rowidcs[1:maxrows],
            colidcs[nextcolumn:nextcolumn],
        )
    end

    # conv is true until convergence is reached
    npivot, conv = aca.convergence(rowbuffer, colbuffer, npivot, maxrows, maxcolumns, tol)

    while conv && npivot < maxrank
        npivot += 1
        @views nextrow = aca.rowpivoting(colbuffer[1:maxrows, npivot - 1])
        push!(rows, nextrow)
        nextrc!(
            view(rowbuffer, npivot, 1:maxcolumns),
            A,
            rowidcs[nextrow],
            colidcs[1:maxcolumns],
        )

        for k in 1:(npivot - 1)
            for kk in 1:maxcolumns
                rowbuffer[npivot, kk] -= colbuffer[nextrow, k] * rowbuffer[k, kk]
            end
        end

        @views nextcolumn = aca.columnpivoting(rowbuffer[npivot, 1:maxcolumns])
        push!(cols, nextcolumn)
        denominator = rowbuffer[npivot, nextcolumn]
        if denominator != 0.0
            @. rowbuffer[npivot, 1:maxcolumns] /= denominator
            nextrc!(
                view(colbuffer, 1:maxrows, npivot:npivot),
                A,
                rowidcs[1:maxrows],
                colidcs[nextcolumn],
            )
        end

        for k in 1:(npivot - 1)
            for kk in 1:maxrows
                colbuffer[kk, npivot] -= colbuffer[kk, k] * rowbuffer[k, nextcolumn]
            end
        end

        npivot, conv = aca.convergence(
            rowbuffer, colbuffer, npivot, maxrows, maxcolumns, tol
        )
    end

    return rows, cols, npivot
end
#=
function aca(
    M::LazyMatrix{I,K};
    rowpivoting::PivStrat=MaximumValue(zeros(Bool, length(M.τ))),
    columnpivoting::PivStrat=MaximumValue(zeros(Bool, length(M.σ))),
    convergence::ConvCrit=Default(0.0),
    maxrank=Int(round(length(M.τ) * length(M.σ) / (length(M.τ) + length(M.σ)))),
    tol=1e-4,
    svdrecompress=false,
) where {I,K}
    compressor = ACA(rowpivoting, columnpivoting, convergence)
    rowbuffer = zeros(K, maxrank, length(M.σ))
    colbuffer = zeros(K, length(M.τ), maxrank)

    npivots = compressor(M, rowbuffer, colbuffer, maxrank, tol)

    if svdrecompress
        @views Q, R = qr(colbuffer[1:size(M, 1), 1:npivots])
        @views U, s, V = svd(R * rowbuffer[1:npivots, 1:size(M, 2)])

        opt_r = length(s)
        for i in eachindex(s)
            if s[i] < tol * s[1]
                opt_r = i
                break
            end
        end

        A = (Q * U)[1:size(M, 1), 1:opt_r]
        B = (diagm(s) * V')[1:opt_r, 1:size(M, 2)]

        return A, B
    else
        return rows(compressor),
        cols(compressor), colbuffer[1:size(M, 1), 1:npivots],
        rowbuffer[1:npivots, 1:size(M, 2)]
    end
end
=#
