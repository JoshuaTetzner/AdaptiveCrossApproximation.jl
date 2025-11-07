"""
    ACAᵀ{RowPivType,ColPivType,ConvCritType}

Column-first variant of adaptive cross approximation.
Starts by selecting columns first, then rows. Dual of standard ACA.

# Fields

  - `rowpivoting::RowPivType`: Strategy for selecting row pivots
  - `columnpivoting::ColPivType`: Strategy for selecting column pivots
  - `convergence::ConvCritType`: Convergence criterion
"""
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

"""
    ACAᵀ(; rowpivoting=MaximumValue(), columnpivoting=MaximumValue(), convergence=FNormEstimator(0.0, 1e-4))

Construct column-first ACA compressor with specified strategies.

# Arguments

  - `rowpivoting`: Row pivoting strategy (default: `MaximumValue()`)
  - `columnpivoting`: Column pivoting strategy (default: `MaximumValue()`)
  - `convergence`: Convergence criterion (default: `FNormEstimator(0.0, 1e-4)`)
"""
function ACAᵀ(;
    rowpivoting=MaximumValue(),
    columnpivoting=MaximumValue(),
    convergence=FNormEstimator(0.0, 1e-4),
)
    return ACAᵀ(rowpivoting, columnpivoting, convergence)
end

"""
    (aca::ACAᵀ)(rowidcs::AbstractArray{Int}, colidcs::AbstractArray{Int})

Initialize ACAᵀ functor with index sets.
Creates functors for pivoting strategies bound to specific index ranges.

# Arguments

  - `rowidcs::AbstractArray{Int}`: Row indices for this compression
  - `colidcs::AbstractArray{Int}`: Column indices for this compression
"""
function (aca::ACAᵀ)(rowidcs::AbstractArray{Int}, colidcs::AbstractArray{Int})
    return ACAᵀ(aca.rowpivoting(rowidcs), aca.columnpivoting(colidcs), aca.convergence())
end

"""
    (aca::ACAᵀ)(A, rowbuffer, colbuffer, maxrank; rows, cols, rowidcs, colidcs)

Perform column-first ACA compression.
Computes low-rank approximation A ≈ colbuffer * rowbuffer by iteratively selecting columns then rows.

# Arguments

  - `A`: Matrix to compress
  - `rowbuffer::AbstractMatrix{K}`: Pre-allocated row storage (maxrank × ncols)
  - `colbuffer::AbstractMatrix{K}`: Pre-allocated column storage (nrows × maxrank)
  - `maxrank::Int`: Maximum number of pivots
  - `rows`: Selected row indices (optional, pre-allocated)
  - `cols`: Selected column indices (optional, pre-allocated)
  - `rowidcs`: Active row index range (optional)
  - `colidcs`: Active column index range (optional)

# Returns

  - `npivot::Int`: Number of pivots computed
"""
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

"""
    acaᵀ(M; tol=1e-4, rowpivoting, columnpivoting, convergence, maxrank=40)

Convenience function for column-first ACA compression.
Automatically allocates buffers and performs compression.

# Arguments

  - `M::AbstractMatrix{K}`: Matrix to compress
  - `tol`: Convergence tolerance (default: `1e-4`)
  - `rowpivoting`: Row pivoting strategy (default: `MaximumValueFunctor`)
  - `columnpivoting`: Column pivoting strategy (default: `MaximumValueFunctor`)
  - `convergence`: Convergence criterion (default: `FNormEstimator(0.0, tol)`)
  - `maxrank`: Maximum rank (default: `40`)

# Returns

  - `colbuffer`: Column factor (nrows × npivots)
  - `rowbuffer`: Row factor (npivots × ncols)
"""
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
