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
    rowpivoting=MaximumValue(),
    columnpivoting=MaximumValue(),
    convergence=FNormEstimator(0.0, 1e-4),
)
    return ACA(rowpivoting, columnpivoting, convergence)
end

function (aca::ACA)(
    K::AbstractMatrix, rowidcs::AbstractArray{Int}, colidcs::AbstractArray{Int}
)
    return ACA(aca.rowpivoting(rowidcs), aca.columnpivoting(colidcs), aca.convergence())
end

function (aca::ACA{PS,PS,CC})(
    K::AbstractMatrix, rowidcs::AbstractArray{Int}, colidcs::AbstractArray{Int}
) where {PS<:PivStrat,CC<:RandomSampling}
    convergence = aca.convergence(K, rowidcs, colidcs)
    rowpivoting = if aca.rowpivoting isa RandomSamplingPivoting
        aca.rowpivoting(convergence)
    else
        aca.rowpivoting(rowidcs)
    end
    columnpivoting = if aca.columnpivoting isa RandomSamplingPivoting
        aca.columnpivoting(convergence)
    else
        aca.columnpivoting(colidcs)
    end

    return ACA(rowpivoting, columnpivoting, convergence)
end

function (aca::ACA{RP,CP,CC})(
    K::AbstractMatrix, rowidcs::AbstractArray{Int}, colidcs::AbstractArray{Int}
) where {RP,CP,CC<:CombinedConvCrit}
    convergence = aca.convergence(K, rowidcs, colidcs)
    rowpivoting = if aca.rowpivoting isa CombinedPivStrat
        aca.rowpivoting(convergence, rowidcs)
    else
        aca.rowpivoting(rowidcs)
    end
    columnpivoting = if aca.columnpivoting isa CombinedPivStrat
        aca.columnpivoting(convergence, colidcs)
    else
        aca.columnpivoting(colidcs)
    end

    return ACA(rowpivoting, columnpivoting, convergence)
end

function (aca::ACA)(
    A,
    rowbuffer::AbstractMatrix{K},
    colbuffer::AbstractMatrix{K},
    maxrank::Int;
    rowidcs = Vector(1:size(colbuffer, 1)),
    colidcs = Vector(1:size(rowbuffer, 2)),
) where {K}
    rows = Int[1]
    cols = Int[]
    maxrows = size(colbuffer, 1)
    maxcolumns = size(rowbuffer, 2)
    npivot = 1
    nextrow = aca.rowpivoting()
    nextrc!(
        view(rowbuffer, npivot:npivot, 1:maxcolumns),
        A,
        view(rowidcs, 1:1),
        view(colidcs, 1:maxcolumns),
    )
    @views nextcolumn = aca.columnpivoting(rowbuffer[npivot, 1:maxcolumns])
    push!(cols, nextcolumn)
    if rowbuffer[npivot, nextcolumn] != 0.0
        view(rowbuffer, npivot, 1:maxcolumns) ./= view(rowbuffer, npivot, nextcolumn)
    end
    nextrc!(
        view(colbuffer, 1:maxrows, npivot:npivot),
        A,
        view(rowidcs, 1:maxrows),
        view(colidcs, nextcolumn:nextcolumn),
    )

    # conv is true until convergence is reached
    npivot, conv = aca.convergence(rowbuffer, colbuffer, npivot, maxrows, maxcolumns)

    while conv && npivot < maxrank
        npivot += 1
        @views nextrow = aca.rowpivoting(colbuffer[1:maxrows, max(1, npivot - 1)])
        length(rows) < npivot ? push!(rows, nextrow) : rows[npivot] = nextrow
        nextrc!(
            view(rowbuffer, npivot:npivot, 1:maxcolumns),
            A,
            view(rowidcs, nextrow:nextrow),
            view(colidcs, 1:maxcolumns),
        )

        for k in 1:(npivot - 1)
            for kk in 1:maxcolumns
                rowbuffer[npivot, kk] -= colbuffer[nextrow, k] * rowbuffer[k, kk]
            end
        end

        @views nextcolumn = aca.columnpivoting(rowbuffer[npivot, 1:maxcolumns])
        length(cols) < npivot ? push!(cols, nextcolumn) : cols[npivot] = nextcolumn
        if rowbuffer[npivot, nextcolumn] != 0.0
            view(rowbuffer, npivot, 1:maxcolumns) ./= view(rowbuffer, npivot, nextcolumn)
            nextrc!(
                view(colbuffer, 1:maxrows, npivot:npivot),
                A,
                view(rowidcs, 1:maxrows),
                view(colidcs, nextcolumn:nextcolumn),
            )
        end

        for k in 1:(npivot - 1)
            for kk in 1:maxrows
                colbuffer[kk, npivot] -= colbuffer[kk, k] * rowbuffer[k, nextcolumn]
            end
        end

        npivot, conv = aca.convergence(rowbuffer, colbuffer, npivot, maxrows, maxcolumns)
    end

    return npivot, rows, cols
end

function aca(
    M::AbstractMatrix{K};
    tol=1e-4,
    rowpivoting=MaximumValue(zeros(Bool, size(M, 1))),
    columnpivoting=MaximumValue(zeros(Bool, size(M, 1))),
    convergence=FNormEstimator(0.0, tol),
    maxrank=40,
    svdrecompress=false,
) where {K}
    compressor = ACA(rowpivoting, columnpivoting, convergence)
    rowbuffer = zeros(K, maxrank, size(M, 2))
    colbuffer = zeros(K, size(M, 1), maxrank)

    npivots, rows, cols = compressor(M, rowbuffer, colbuffer, maxrank)
    if svdrecompress
        @views Q, R = qr(colbuffer[1:size(M, 1), 1:npivots])
        @views U, s, V = svd(R * rowbuffer[1:npivots, 1:size(M, 2)])

        opt_r = length(s)
        for i in eachindex(s)
            if s[i] < tolerance(convergence) * s[1]
                opt_r = i
                break
            end
        end

        A = (Q * U)[1:size(M, 1), 1:opt_r]
        B = (diagm(s) * V')[1:opt_r, 1:size(M, 2)]

        return A, B
    else
        return colbuffer[1:size(M, 1), 1:npivots], rowbuffer[1:npivots, 1:size(M, 2)]
    end
end
