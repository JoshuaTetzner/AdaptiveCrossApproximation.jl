"""
    iACA{RowPivType,ColPivType,ConvCritType}

Incomplete Adaptive Cross Approximation (iACA) compressor.

Unlike standard ACA, iACA computes only half of the factorization.
It uses geometric pivoting strategies (e.g., mimicry or tree mimicry) to select row or column
pivots based solely on spatial information, making it super efficient for hierarchical matrix
construction where only row or column samples are requiered.

# Fields

  - `rowpivoting::RowPivType`: Strategy for selecting row pivots (geometric)
  - `columnpivoting::ColPivType`: Strategy for selecting column pivots
  - `convergence::ConvCritType`: Convergence criterion
"""
mutable struct iACA{RowPivType,ColPivType,ConvCritType}
    rowpivoting::RowPivType
    columnpivoting::ColPivType
    convergence::ConvCritType

    function iACA(rowpivoting, columnpivoting, convergence)
        return new{typeof(rowpivoting),typeof(columnpivoting),typeof(convergence)}(
            rowpivoting, columnpivoting, convergence
        )
    end
end

"""
    iACA(tpos::Vector{SVector{D,F}}, spos::Vector{SVector{D,F}})

Construct an iACA compressor with default settings for geometric pivoting.

Creates an iACA using maximum value for row pivoting, mimicry pivoting for columns
(mimicking the spatial distribution of a fully pivoting when selecting from `spos`), and Frobenius
norm extrapolation for convergence.

# Arguments

  - `tpos`: Test/target point positions (reference distribution)
  - `spos`: Source point positions (candidates for selection)
"""
function iACA(tpos::Vector{SVector{D,F}}, spos::Vector{SVector{D,F}}) where {D,F<:Real}
    return iACA(
        MaximumValue(),
        MimicryPivoting(tpos, spos),
        FNormExtrapolator(iFNormEstimator(F(1e-4))),
    )
end

"""
    (iaca::iACA{GeoPivStrat,ValuePivStrat,ConvCrit})(rows, cols)

Initialize iACA functor for row matrix compression with geometric row pivoting.
Creates functors for geometric row pivoting and value-based column pivoting.

# Arguments

  - `rows::AbstractVector{Int}`: Row indices for geometric pivoting
  - `cols::AbstractVector{Int}`: Column indices for geometric pivoting
"""
function (iaca::iACA{RowPivType,ColPivType,ConvCritType})(
    rows::AbstractVector{Int}, cols::AbstractVector{Int}
) where {RowPivType<:GeoPivStrat,ColPivType<:ValuePivStrat,ConvCritType<:ConvCrit}
    return iACA(iaca.rowpivoting(cols, rows), iaca.columnpivoting(cols), iaca.convergence())
end

"""
    (iaca::iACA{TreeMimicryPivoting,ValuePivStrat,ConvCrit})(Ft, cols, maxrank)

Initialize iACA functor for tree-based row pivoting.
For hierarchical matrices where row selection uses tree-aware mimicry.

# Arguments

  - `Ft::AbstractVector{Int}`: Tree structure for row pivoting
  - `cols::AbstractVector{Int}`: Column indices
  - `maxrank::Int`: Maximum rank for approximation
"""
function (iaca::iACA{RowPivType,ColPivType,ConvCritType})(
    Fs::AbstractVector{Int}, cols::AbstractVector{Int}, maxrank::Int
) where {RowPivType<:TreeMimicryPivoting,ColPivType<:ValuePivStrat,ConvCritType<:ConvCrit}
    return iACA(
        iaca.rowpivoting(Fs, cols, maxrank), iaca.columnpivoting(cols), iaca.convergence()
    )
end

"""
    (iaca::iACA{MimicryPivoting,ValuePivStrat,ConvCrit})(A, colbuffer, rowbuffer, maxrank; kwargs...)

Convenience method delegating to main computational routine for mimicry-based row pivoting.

# Arguments

  - `A`: Matrix to compress
  - `colbuffer::AbstractMatrix{K}`: Buffer for column data
  - `rowbuffer::AbstractMatrix{K}`: Buffer for row data
  - `maxrank::Int`: Maximum rank
  - `rows`: Row indices (optional keyword)
  - `cols`: Column indices (optional keyword)
  - `rowidcs`: Row index range (optional keyword)
  - `colidcs`: Column index range (optional keyword)
"""
function (iaca::iACA{RowPivType,ColPivType,ConvCritType})(
    A,
    rowbuffer::AbstractMatrix{K},
    colbuffer::AbstractMatrix{K},
    maxrank::Int;
    rows=zeros(Int, maxrank),
    cols=zeros(Int, maxrank),
    rowidcs=Vector(1:size(A, 1)),
    colidcs=Vector(1:size(A, 2)),
) where {K,RowPivType<:MimicryPivoting,ColPivType<:ValuePivStrat,ConvCritType<:ConvCrit}
    return iaca(rowidcs, colidcs)(A, rowbuffer, colbuffer, maxrank, rows, cols, colidcs)
end

"""
    (iaca::iACA{TreeMimicryPivoting,ValuePivStrat,ConvCrit})(A, colbuffer, rowbuffer, maxrank; kwargs...)

Convenience method delegating to main computational routine for tree-based row pivoting.

# Arguments

  - `A`: Matrix to compress
  - `colbuffer::AbstractMatrix{K}`: Buffer for column data
  - `rowbuffer::AbstractMatrix{K}`: Buffer for row data
  - `maxrank::Int`: Maximum rank
  - `rows`: Row indices (optional keyword)
  - `cols`: Column indices (optional keyword)
  - `rowidcs`: Row index range (optional keyword)
  - `colidcs`: Column index range (optional keyword)
"""
function (iaca::iACA{RowPivType,ColPivType,ConvCritType})(
    A,
    rowbuffer::AbstractMatrix{K},
    colbuffer::AbstractMatrix{K},
    maxrank::Int;
    rows=zeros(Int, maxrank),
    cols=zeros(Int, maxrank),
    rowidcs=Vector(1:size(A, 1)),
    colidcs=Vector(1:size(A, 2)),
) where {K,RowPivType<:TreeMimicryPivoting,ColPivType<:ValuePivStrat,ConvCritType<:ConvCrit}
    return iaca(rowidcs, colidcs, maxrank)(
        A, rowbuffer, colbuffer, maxrank, rows, cols, colidcs
    )
end

"""
    (iaca::iACA{GeoPivStratFunctor,ValuePivStratFunctor,ConvCritFunctor})(A, colbuffer, rowbuffer, maxrank, rows, cols, colidcs)

Main computational routine for row matrix iACA (geometric row pivoting, value-based column pivoting).
Performs incomplete ACA compression where rows are selected geometrically and columns by maximum value.

# Arguments

  - `A`: Matrix to compress
  - `colbuffer::AbstractMatrix{K}`: Buffer for column data
  - `rowbuffer::AbstractMatrix{K}`: Buffer for row data
  - `maxrank::Int`: Maximum rank
  - `rows::Vector{Int}`: Row indices storage
  - `cols::Vector{Int}`: Column indices storage
  - `colidcs::Vector{Int}`: Column index range

# Returns

  - `npivot::Int`: Number of pivots computed
  - `rows::Vector{Int}`: Selected row indices
  - `cols::Vector{Int}`: Selected column indices (global)
"""
function (iaca::iACA{RowPivType,ColPivType,ConvCritType})(
    A,
    colbuffer::AbstractMatrix{K},
    rowbuffer::AbstractMatrix{K},
    maxrank::Int,
    rows::T,
    cols::T,
    #rowidcs::T,
    colidcs::T,
) where {
    K,
    RowPivType<:GeoPivStratFunctor,
    ColPivType<:ValuePivStratFunctor,
    ConvCritType<:ConvCritFunctor,
    T<:Vector{Int},
}
    maxcolumn = length(colidcs)
    npivot = 1

    rows[npivot] = iaca.rowpivoting()
    nextrc!(
        view(rowbuffer, npivot:npivot, 1:maxcolumn),
        A,
        view(rows, npivot:npivot),
        view(colidcs, 1:maxcolumn),
    )
    normF!(iaca.convergence.estimator, rowbuffer[npivot, 1:maxcolumn], npivot)
    colbuffer[1, 1] = K(1.0)
    cols[npivot] = iaca.columnpivoting(rowbuffer[npivot, 1:maxcolumn])

    npivot, conv = iaca.convergence(rowbuffer[npivot, 1:maxcolumn], npivot)

    while conv && npivot < maxrank
        npivot += 1

        rows[npivot] = iaca.rowpivoting(npivot)
        nextrc!(
            view(rowbuffer, npivot:npivot, 1:maxcolumn),
            A,
            view(rows, npivot:npivot),
            view(colidcs, 1:maxcolumn),
        )

        # Norm update
        normF!(iaca.convergence.estimator, rowbuffer[npivot, 1:maxcolumn], npivot)

        colbuffer[npivot, npivot] = K(1.0)
        for k in 1:(npivot - 1)
            @views colbuffer[npivot, k] =
                rowbuffer[k, cols[k]]^-1 * rowbuffer[npivot, cols[k]]
            for kk in 1:maxcolumn
                @views rowbuffer[npivot, kk] -= rowbuffer[k, kk] * colbuffer[npivot, k]
            end
        end
        cols[npivot] = iaca.columnpivoting(rowbuffer[npivot, 1:maxcolumn])
        npivot, conv = iaca.convergence(rowbuffer[npivot, 1:maxcolumn], npivot)
    end

    return npivot, rows[1:npivot], colidcs[cols[1:npivot]]
end

# ColumnMatrix

"""
    (iaca::iACA{ValuePivStrat,GeoPivStrat,ConvCrit})(rows, cols)

Initialize iACA functor for column matrix compression with geometric column pivoting.
Creates functors for value-based row pivoting and geometric column pivoting.

# Arguments

  - `rows::AbstractVector{Int}`: Row indices
  - `cols::AbstractVector{Int}`: Column indices for geometric pivoting
"""
function (iaca::iACA{RowPivType,ColPivType,ConvCritType})(
    rows::AbstractVector{Int}, cols::AbstractVector{Int}
) where {RowPivType<:ValuePivStrat,ColPivType<:GeoPivStrat,ConvCritType<:ConvCrit}
    return iACA(iaca.rowpivoting(rows), iaca.columnpivoting(rows, cols), iaca.convergence())
end

"""
    (iaca::iACA{ValuePivStrat,TreeMimicryPivoting,ConvCrit})(rows, Ft, maxrank)

Initialize iACA functor for tree-based column pivoting.
For hierarchical matrices where column selection uses tree-aware mimicry.

# Arguments

  - `rows::AbstractVector{Int}`: Row indices
  - `Ft::AbstractVector{Int}`: Tree structure for column pivoting
  - `maxrank::Int`: Maximum rank for approximation
"""
function (iaca::iACA{RowPivType,ColPivType,ConvCritType})(
    rows::AbstractVector{Int}, Ft::AbstractVector{Int}, maxrank::Int
) where {RowPivType<:ValuePivStrat,ColPivType<:TreeMimicryPivoting,ConvCritType<:ConvCrit}
    return iACA(
        iaca.rowpivoting(rows), iaca.columnpivoting(Ft, rows, maxrank), iaca.convergence()
    )
end

"""
    (iaca::iACA{ValuePivStrat,MimicryPivoting,ConvCrit})(A, rowbuffer, colbuffer, maxrank; kwargs...)

Convenience method delegating to main computational routine for mimicry-based column pivoting.

# Arguments

  - `A`: Matrix to compress
  - `rowbuffer::AbstractArray{K}`: Buffer for row data
  - `colbuffer::AbstractArray{K}`: Buffer for column data
  - `maxrank::Int`: Maximum rank
  - `rows`: Row indices (optional keyword)
  - `cols`: Column indices (optional keyword)
  - `rowidcs`: Row index range (optional keyword)
  - `colidcs`: Column index range (optional keyword)
"""
function (iaca::iACA{RowPivType,ColPivType,ConvCritType})(
    A,
    rowbuffer::AbstractArray{K},
    colbuffer::AbstractArray{K},
    maxrank::Int;
    rows=zeros(Int, maxrank),
    cols=zeros(Int, maxrank),
    rowidcs=Vector(1:size(A, 1)),
    colidcs=Vector(1:size(A, 2)),
) where {K,RowPivType<:ValuePivStrat,ColPivType<:MimicryPivoting,ConvCritType<:ConvCrit}
    return iaca(rowidcs, colidcs)(A, rowbuffer, colbuffer, maxrank, rows, cols, rowidcs)
end

"""
    (iaca::iACA{ValuePivStrat,TreeMimicryPivoting,ConvCrit})(A, rowbuffer, colbuffer, maxrank; kwargs...)

Convenience method delegating to main computational routine for tree-based column pivoting.

# Arguments

  - `A`: Matrix to compress
  - `rowbuffer::AbstractArray{K}`: Buffer for row data
  - `colbuffer::AbstractArray{K}`: Buffer for column data
  - `maxrank::Int`: Maximum rank
  - `rows`: Row indices (optional keyword)
  - `cols`: Column indices (optional keyword)
  - `rowidcs`: Row index range (optional keyword)
  - `colidcs`: Column index range (optional keyword)
"""
function (iaca::iACA{RowPivType,ColPivType,ConvCritType})(
    A,
    rowbuffer::AbstractArray{K},
    colbuffer::AbstractArray{K},
    maxrank::Int;
    rows=zeros(Int, maxrank),
    cols=zeros(Int, maxrank),
    rowidcs=Vector(1:size(A, 1)),
    colidcs=Vector(1:size(A, 2)),
) where {K,RowPivType<:ValuePivStrat,ColPivType<:TreeMimicryPivoting,ConvCritType<:ConvCrit}
    return iaca(rowidcs, colidcs, maxrank)(
        A, rowbuffer, colbuffer, maxrank, rows, cols, rowidcs
    )
end

"""
    (iaca::iACA{ValuePivStratFunctor,GeoPivStratFunctor,ConvCritFunctor})(A, colbuffer, rowbuffer, maxrank, rows, cols, rowidcs)

Main computational routine for column matrix iACA (value-based row pivoting, geometric column pivoting).
Performs incomplete ACA compression where columns are selected geometrically and rows by maximum value.

# Arguments

  - `A`: Matrix to compress
  - `colbuffer::AbstractArray{K}`: Buffer for column data
  - `rowbuffer::AbstractArray{K}`: Buffer for row data
  - `maxrank::Int`: Maximum rank
  - `rows::Vector{Int}`: Row indices storage
  - `cols::Vector{Int}`: Column indices storage
  - `rowidcs::Vector{Int}`: Row index range

# Returns

  - `npivot::Int`: Number of pivots computed
  - `rows::Vector{Int}`: Selected row indices (global)
  - `cols::Vector{Int}`: Selected column indices
"""
function (iaca::iACA{RowPivType,ColPivType,ConvCritType})(
    A,
    colbuffer::AbstractArray{K},
    rowbuffer::AbstractArray{K},
    maxrank::Int,
    rows::T,
    cols::T,
    rowidcs::T,
    #colidcs::T,
) where {
    K,
    RowPivType<:ValuePivStratFunctor,
    ColPivType<:GeoPivStratFunctor,
    ConvCritType<:ConvCritFunctor,
    T<:Vector{Int},
}
    maxrow = length(rowidcs)
    npivot = 1

    cols[npivot] = iaca.columnpivoting()
    nextrc!(
        view(colbuffer, 1:maxrow, npivot:npivot),
        A,
        view(rowidcs, 1:maxrow),
        view(cols, npivot:npivot),
    )
    normF!(iaca.convergence.estimator, colbuffer[1:maxrow, npivot], npivot)
    rowbuffer[1, 1] = K(1.0)
    rows[npivot] = iaca.rowpivoting(colbuffer[1:maxrow, npivot])

    npivot, conv = iaca.convergence(colbuffer[1:maxrow, npivot], npivot)

    while conv && npivot < maxrank
        npivot += 1

        cols[npivot] = iaca.columnpivoting(npivot)

        nextrc!(
            view(colbuffer, 1:maxrow, npivot:npivot),
            A,
            view(rowidcs, 1:maxrow),
            view(cols, npivot:npivot),
        )

        # Norm update
        normF!(iaca.convergence.estimator, colbuffer[1:maxrow, npivot], npivot)

        rowbuffer[npivot, npivot] = K(1.0)
        for k in 1:(npivot - 1)
            @views rowbuffer[k, npivot] =
                colbuffer[rows[k], k]^-1 * colbuffer[rows[k], npivot]
            for kk in 1:maxrow
                @views colbuffer[kk, npivot] -= colbuffer[kk, k] * rowbuffer[k, npivot]
            end
        end
        rows[npivot] = iaca.rowpivoting(colbuffer[1:maxrow, npivot])
        npivot, conv = iaca.convergence(colbuffer[1:maxrow, npivot], npivot)
    end

    return npivot, rowidcs[rows[1:npivot]], cols[1:npivot]
end
