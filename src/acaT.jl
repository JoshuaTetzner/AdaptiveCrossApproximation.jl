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
    ACAᵀ(; rowpivoting=MaximumValue(), columnpivoting=MaximumValue(), convergence=FNormEstimator(1e-4))

Construct column-first ACA compressor with specified strategies.

# Arguments

  - `rowpivoting`: Row pivoting strategy (default: `MaximumValue()`)
  - `columnpivoting`: Column pivoting strategy (default: `MaximumValue()`)
  - `convergence`: Convergence criterion (default: `FNormEstimator(1e-4)`)
"""
function ACAᵀ(;
    rowpivoting=MaximumValue(),
    columnpivoting=MaximumValue(),
    convergence=FNormEstimator(1e-4),
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
    (aca::ACAᵀ{P,P,C})(A, colbuffer, rowbuffer, maxrank; kwargs...)

Convenience method that initializes pivoting functors when using uniform strategies.

Delegates to the main computational routine after creating index-specialized functors.
Only available when both pivoting strategies are of the same stateless type `P <: PivStrat`.

See the main `(aca::ACAᵀ)(A, colbuffer, rowbuffer, rows, cols, rowidcs, colidcs, maxrank)`
method for detailed argument documentation.
"""
function (aca::ACAᵀ{P,P,C})(
    A,
    colbuffer::AbstractArray{K},
    rowbuffer::AbstractArray{K},
    maxrank::Int;
    rows=zeros(Int, maxrank),
    cols=zeros(Int, maxrank),
    rowidcs=Vector(1:size(colbuffer, 1)),
    colidcs=Vector(1:size(rowbuffer, 2)),
) where {K,P<:PivStrat,C<:ConvCrit}
    return aca(rowidcs, colidcs)(
        A, colbuffer, rowbuffer, rows, cols, rowidcs, colidcs, maxrank
    )
end

"""
    (aca::ACAᵀ)(A, colbuffer, rowbuffer, maxrank; rows, cols, rowidcs, colidcs)

Perform column-first ACA compression.
Computes low-rank approximation A ≈ colbuffer * rowbuffer by iteratively selecting columns then rows.

# Arguments

  - `A`: Matrix to compress
  - `colbuffer::AbstractArray{K}`: Pre-allocated column storage (nrows × maxrank)
  - `rowbuffer::AbstractArray{K}`: Pre-allocated row storage (maxrank × ncols)
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
    colbuffer::AbstractArray{K},
    rowbuffer::AbstractArray{K},
    rows::T,
    cols::T,
    rowidcs::T,
    colidcs::T,
    maxrank::Int,
) where {K,T<:Vector{Int}}
    maxrows = size(colbuffer, 1)
    maxcols = size(rowbuffer, 2)
    npivot = 1
    nextcol = aca.columnpivoting()
    cols[1] = colidcs[nextcol]
    nextrc!(
        view(colbuffer, 1:maxrows, npivot:npivot),
        A,
        view(rowidcs, 1:maxrows),
        view(colidcs, 1:1),
    )
    @views nextrow = aca.rowpivoting(colbuffer[1:maxrows, npivot])
    rows[npivot] = rowidcs[nextrow]
    if colbuffer[nextrow, npivot] != 0.0
        view(colbuffer, 1:maxrows, npivot) ./= view(colbuffer, nextrow, npivot)
    end
    nextrc!(
        view(rowbuffer, npivot:npivot, 1:maxcols),
        A,
        view(rowidcs, nextrow:nextrow),
        view(colidcs, 1:maxcols),
    )

    # conv is true until convergence is reached
    npivot, conv = aca.convergence(rowbuffer, colbuffer, npivot, maxrows, maxcols)

    while conv && npivot < maxrank
        npivot += 1
        @views nextcol = aca.columnpivoting(rowbuffer[max(1, npivot - 1), 1:maxcols])
        cols[npivot] = colidcs[nextcol]
        nextrc!(
            view(colbuffer, 1:maxrows, npivot:npivot),
            A,
            view(rowidcs, 1:maxrows),
            view(colidcs, nextcol:nextcol),
        )

        for k in 1:(npivot - 1)
            for kk in 1:maxrows
                colbuffer[kk, npivot] -= rowbuffer[k, nextcol] * colbuffer[kk, k]
            end
        end

        @views nextrow = aca.rowpivoting(colbuffer[1:maxrows, npivot])
        rows[npivot] = rowidcs[nextrow]
        if colbuffer[nextrow, npivot] != 0.0
            view(colbuffer, 1:maxrows, npivot) ./= view(colbuffer, nextrow, npivot)
            nextrc!(
                view(rowbuffer, npivot:npivot, 1:maxcols),
                A,
                view(rowidcs, nextrow:nextrow),
                view(colidcs, 1:maxcols),
            )
        end

        for k in 1:(npivot - 1)
            for kk in 1:maxcols
                rowbuffer[npivot, kk] -= rowbuffer[k, kk] * colbuffer[nextrow, k]
            end
        end

        npivot, conv = aca.convergence(rowbuffer, colbuffer, npivot, maxrows, maxcols)
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
    rowpivoting=MaximumValue(),
    columnpivoting=MaximumValue(),
    convergence=FNormEstimator(tol),
    maxrank::Int=40,
    svdrecompress=false,
) where {K}
    compressor = ACAᵀ(rowpivoting, columnpivoting, convergence)
    rowbuffer = zeros(K, maxrank, size(M, 2))
    colbuffer = zeros(K, size(M, 1), maxrank)

    npivots = compressor(M, colbuffer, rowbuffer, maxrank)

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
