# ACA starting with a column
struct ACAᵀ{RowPivType,ColPivType,ConvCritType}
    rowpivoting::RowPivType
    columnpivoting::ColPivType
    convergence::ConvCritType

    function ACAᵀ(rowpivoting, columnpivoting, convergence)
        return new{typeof(rowpivoting),typeof(columnpivoting),typeof(convergence)}(
            rowpivoting, columnpivoting, convergence
        )
    end
end

function ACAᵀ(;
    rowpivoting=MaximumValue(),
    columnpivoting=MaximumValue(),
    convergence=FNormEstimator(0.0, 1e-4),
)
    return ACAᵀ(rowpivoting, columnpivoting, convergence)
end

function (aca::ACAᵀ)(rowidcs::AbstractArray{Int}, colidcs::AbstractArray{Int})
    return ACAᵀ(aca.rowpivoting(rowidcs), aca.columnpivoting(colidcs), aca.convergence())
end

function (aca::ACAᵀ)(
    A,
    rowbuffer::AbstractMatrix{K},
    colbuffer::AbstractMatrix{K},
    maxrank::Int;
    rows=zeros(Int, maxrank),
    cols=zeros(Int, maxrank),
    rowidcs=Vector(1:size(colbuffer, 1)),
    colidcs=Vector(1:size(rowbuffer, 2)),
) where {K}
    maxrows = size(colbuffer, 1)
    maxcolumns = size(rowbuffer, 2)
    npivot = 1
    cols[1] = aca.columnpivoting()
    nextrc!(
        view(colbuffer, 1:maxrows, npivot:npivot),
        A,
        view(rowidcs, 1:maxrows),
        view(colidcs, 1:1),
    )
    @views rows[npivot] = aca.rowpivoting(colbuffer[1:maxrows, npivot])
    if colbuffer[rows[npivot], npivot] != 0.0
        view(colbuffer, 1:maxrows, npivot) ./= view(colbuffer, rows[npivot], npivot)
    end
    nextrc!(
        view(rowbuffer, npivot:npivot, 1:maxcolumns),
        A,
        view(rowidcs, rows[npivot]:rows[npivot]),
        view(colidcs, 1:maxcolumns),
    )

    # conv is true until convergence is reached
    npivot, conv = aca.convergence(rowbuffer, colbuffer, npivot, maxrows, maxcolumns)

    while conv && npivot < maxrank
        npivot += 1
        @views cols[npivot] = aca.columnpivoting(rowbuffer[max(1, npivot - 1), 1:maxcols])
        nextrc!(
            view(colbuffer, 1:maxrows, npivot:npivot),
            A,
            view(rowidcs, 1:maxrows),
            view(colidcs, cols[npivot]:cols[npivot]),
        )

        for k in 1:(npivot - 1)
            for kk in 1:maxrows
                colbuffer[kk, npivot] -= rowbuffer[k, cols[npivot]] * colbuffer[kk, k]
            end
        end

        @views rows[npivot] = aca.rowpivoting(colbuffer[1:rows, npivot])
        if colbuffer[rows[npivot], npivot] != 0.0
            view(colbuffer, 1:maxrows, npivot) ./= view(colbuffer, rows[npivot], npivot)
            nextrc!(
                view(rowbuffer, npivot:npivot, 1:maxcolumns),
                A,
                view(rowidcs, rows[npivot]:rows[npivot]),
                view(colidcs, 1:maxcolumns),
            )
        end

        for k in 1:(npivot - 1)
            for kk in 1:maxcolumns
                rowbuffer[npivot, kk] -= rowbuffer[k, kk] * colbuffer[rows[npivot], k]
            end
        end

        npivot, conv = aca.convergence(rowbuffer, colbuffer, npivot, maxrows, maxcolumns)
    end

    return npivot
end

function acaᵀ(
    M::AbstractMatrix{K};
    tol=1e-4,
    rowpivoting=MaximumValueFunctor(zeros(Bool, size(M, 1))),
    columnpivoting=MaximumValueFunctor(zeros(Bool, size(M, 1))),
    convergence=FNormEstimator(0.0, tol),
    maxrank=40,
) where {K}
    compressor = ACAᵀ(rowpivoting, columnpivoting, convergence)
    rowbuffer = zeros(K, maxrank, size(M, 2))
    colbuffer = zeros(K, size(M, 1), maxrank)

    npivots = compressor(M, rowbuffer, colbuffer, maxrank)
    return colbuffer[:, 1:npivots], rowbuffer[1:npivots, :]
end
