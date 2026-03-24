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
struct iACA{RowPivType,ColPivType,ConvCritType}
    rowpivoting::RowPivType
    columnpivoting::ColPivType
    convergence::ConvCritType

    function iACA(rowpivoting, columnpivoting, convergence)
        return new{typeof(rowpivoting),typeof(columnpivoting),typeof(convergence)}(
            rowpivoting, columnpivoting, convergence
        )
    end
end

function iACA(tpos::Vector{SVector{D,F}}, spos::Vector{SVector{D,F}}) where {D,F<:Real}
    return iACA(
        MaximumValue(),
        MimicryPivoting(tpos, spos),
        FNormExtrapolator(iFNormEstimator(F(1e-4))),
    )
end

function (iaca::iACA{RowPivType,ColPivType,ConvCritType})(
    rowidcs::AbstractVector{Int}, colidcs::AbstractVector{Int}, maxrank::Int
) where {RowPivType<:GeoPivStrat,ColPivType<:MaximumValue,ConvCritType<:ConvCrit}
    rowpivstrat = _buildpivstrat(iaca.rowpivoting, colidcs, rowidcs, maxrank)
    return iACA(rowpivstrat, iaca.columnpivoting(colidcs), iaca.convergence(maxrank))
end

function reset!(
    iaca::iACA{RP,CP,CC}, rowidcs::AbstractVector{Int}, colidcs::AbstractVector{Int}
) where {RP<:GeoPivStratFunctor,CP<:MaximumValueFunctor,CC<:ConvCritFunctor}
    reset!(iaca.rowpivoting, colidcs, rowidcs)
    reset!(iaca.columnpivoting, colidcs)
    reset!(iaca.convergence)
    return nothing
end

#=
@inline function _run_iaca_stateful(
    iaca::iACA{RowPivType,ColPivType,ConvCritType},
    A,
    colbuffer,
    rowbuffer,
    maxrank::Int,
    rows,
    cols,
    rowidcs,
    colidcs,
) where {
    RowPivType<:GeoPivStratFunctor,
    ColPivType<:ValuePivStratFunctor,
    ConvCritType<:ConvCritFunctor,
}
    return iaca(A, colbuffer, rowbuffer, maxrank, rows, cols, colidcs)
end

@inline function _run_iaca_stateful(
    iaca::iACA{RowPivType,ColPivType,ConvCritType},
    A,
    colbuffer,
    rowbuffer,
    maxrank::Int,
    rows,
    cols,
    rowidcs,
    colidcs,
) where {
    RowPivType<:ValuePivStratFunctor,
    ColPivType<:GeoPivStratFunctor,
    ConvCritType<:ConvCritFunctor,
}
    return iaca(A, colbuffer, rowbuffer, maxrank, rows, cols, rowidcs)
end
=#

#=
function (iaca::iACA{RowPivType,ColPivType,ConvCritType})(
    rows::AbstractVector{Int}, cols::AbstractVector{Int}
) where {
    RowPivType<:MimicryPivotingFunctor,
    ColPivType<:ValuePivStratFunctor,
    ConvCritType<:ConvCritFunctor,
}
    iaca.rowpivoting.refcentroid = _centroid(iaca.rowpivoting.pivoting.refpos, cols)
    return reset!(iaca, rows, cols)
end

function (iaca::iACA{RowPivType,ColPivType,ConvCritType})(
    Fs::AbstractVector{Int}, cols::AbstractVector{Int}; maxrank::Int=40
) where {
    RowPivType<:TreeMimicryPivotingFunctor,
    ColPivType<:ValuePivStratFunctor,
    ConvCritType<:ConvCritFunctor,
}
    iaca.rowpivoting.refcentroid = _centroid(iaca.rowpivoting.pivoting.refpos, cols)
    return reset!(iaca, Fs, cols)
end

function (iaca::iACA{RowPivType,ColPivType,ConvCritType})(
    A,
    colbuffer::AbstractMatrix{K},
    rowbuffer::AbstractMatrix{K},
    maxrank::Int;
    rows=zeros(Int, maxrank),
    cols=zeros(Int, maxrank),
    rowidcs=Vector(1:size(A, 1)),
    colidcs=Vector(1:size(A, 2)),
) where {
    K,RowPivType<:PivStratFunctor,ColPivType<:PivStratFunctor,ConvCritType<:ConvCritFunctor
}
    stateful = iaca(rowidcs, colidcs)
    return _run_iaca_stateful(
        stateful, A, colbuffer, rowbuffer, maxrank, rows, cols, rowidcs, colidcs
    )
end

function (iaca::iACA{RowPivType,ColPivType,ConvCritType})(
    A,
    colbuffer::AbstractMatrix{K},
    rowbuffer::AbstractMatrix{K},
    maxrank::Int;
    rows=zeros(Int, maxrank),
    cols=zeros(Int, maxrank),
    rowidcs=Vector(1:size(A, 1)),
    colidcs=Vector(1:size(A, 2)),
) where {K,RowPivType<:MimicryPivoting,ColPivType<:ValuePivStrat,ConvCritType<:ConvCrit}
    return iaca(rowidcs, colidcs)(A, colbuffer, rowbuffer, maxrank, rows, cols, colidcs)
end
=#
function (iaca::iACA{RP,CP,CC})(
    A,
    colbuffer::AbstractMatrix{K},
    rowbuffer::AbstractMatrix{K},
    maxrank::Int;
    rowpivs=zeros(Int, maxrank),
    colpivs=zeros(Int, maxrank),
    rowidcs=Vector(1:size(A, 1)),
    colidcs=Vector(1:size(A, 2)),
) where {K,RP<:GeoPivStratFunctor,CP<:MaximumValueFunctor,CC<:ConvCritFunctor}
    reset!(iaca, rowidcs, colidcs)
    return iaca(rowidcs, colidcs, maxrank)(
        A, colbuffer, rowbuffer, maxrank, rowpivs, colpivs, colidcs
    )
end

function (iaca::iACA{RP,CP,CC})(
    A,
    colbuffer::AbstractArray{K},
    rowbuffer::AbstractArray{K},
    rowpivs::T,
    colpivs::T,
    rowidcs::T,
    colidcs::T,
    maxrank::Int;
) where {
    K,RP<:GeoPivStratFunctor,CP<:MaximumValueFunctor,CC<:ConvCritFunctor,T<:Vector{Int}
}
    reset!(iaca, rowidcs, colidcs)
    return iaca(A, colbuffer, rowbuffer, rowpivs, colpivs, colidcs, maxrank)
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
    rowpivs::T,
    colpivs::T,
    colidcs::T,
    maxrank::Int,
) where {
    K,
    RowPivType<:GeoPivStratFunctor,
    ColPivType<:ValuePivStratFunctor,
    ConvCritType<:ConvCritFunctor,
    T<:Vector{Int},
}
    maxcolumn = length(colidcs)
    npivot = 1

    rowpivs[npivot] = iaca.rowpivoting()
    nextrc!(
        view(rowbuffer, npivot:npivot, 1:maxcolumn),
        A,
        view(rowpivs, npivot:npivot),
        view(colidcs, 1:maxcolumn),
    )
    normF!(iaca.convergence.estimator, rowbuffer[npivot, 1:maxcolumn], npivot)
    colbuffer[1, 1] = K(1.0)
    colpivs[npivot] = iaca.columnpivoting(rowbuffer[npivot, 1:maxcolumn])

    npivot, conv = iaca.convergence(rowbuffer[npivot, 1:maxcolumn], npivot)

    while conv && npivot < maxrank
        npivot += 1

        rowpivs[npivot] = iaca.rowpivoting(npivot)
        nextrc!(
            view(rowbuffer, npivot:npivot, 1:maxcolumn),
            A,
            view(rowpivs, npivot:npivot),
            view(colidcs, 1:maxcolumn),
        )

        # Norm update
        normF!(iaca.convergence.estimator, rowbuffer[npivot, 1:maxcolumn], npivot)

        colbuffer[npivot, npivot] = K(1.0)
        for k in 1:(npivot - 1)
            @views colbuffer[npivot, k] =
                rowbuffer[k, colpivs[k]]^-1 * rowbuffer[npivot, colpivs[k]]
            for kk in 1:maxcolumn
                @views rowbuffer[npivot, kk] -= rowbuffer[k, kk] * colbuffer[npivot, k]
            end
        end
        colpivs[npivot] = iaca.columnpivoting(rowbuffer[npivot, 1:maxcolumn])
        npivot, conv = iaca.convergence(rowbuffer[npivot, 1:maxcolumn], npivot)
    end

    return npivot, rowpivs[1:npivot], colidcs[colpivs[1:npivot]]
end

# ColumnMatrix
#=
"""
    (iaca::iACA{ValuePivStrat,MimicryPivoting,ConvCrit})(rows, cols)

Initialize iACA functor for column matrix compression with geometric column pivoting.
Creates functors for value-based row pivoting and geometric column pivoting.

# Arguments

  - `rows::AbstractVector{Int}`: Row indices
  - `cols::AbstractVector{Int}`: Column indices for geometric pivoting
"""
function (iaca::iACA{RowPivType,ColPivType,ConvCritType})(
    rows::AbstractVector{Int}, cols::AbstractVector{Int}; maxrank::Int=40
) where {RowPivType<:ValuePivStrat,ColPivType<:MimicryPivoting,ConvCritType<:ConvCrit}
    return iACA(
        iaca.rowpivoting(rows), iaca.columnpivoting(rows, cols), iaca.convergence(maxrank)
    )
end

function (iaca::iACA{RowPivType,ColPivType,ConvCritType})(
    rows::AbstractVector{Int}, cols::AbstractVector{Int}; maxrank::Int=40
) where {
    RowPivType<:ValuePivStratFunctor,
    ColPivType<:MimicryPivotingFunctor,
    ConvCritType<:ConvCritFunctor,
}
    iaca.columnpivoting.refcentroid = _centroid(iaca.columnpivoting.pivoting.refpos, rows)
    return reset!(iaca, rows, cols)
end

function (iaca::iACA{RowPivType,ColPivType,ConvCritType})(
    rows::AbstractVector{Int}, Ft::AbstractVector{Int}; maxrank::Int=40
) where {
    RowPivType<:ValuePivStratFunctor,
    ColPivType<:TreeMimicryPivotingFunctor,
    ConvCritType<:ConvCritFunctor,
}
    iaca.columnpivoting.refcentroid = _centroid(iaca.columnpivoting.pivoting.refpos, rows)
    return reset!(iaca, rows, Ft)
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
        iaca.rowpivoting(rows),
        iaca.columnpivoting(Ft, rows, maxrank),
        iaca.convergence(maxrank),
    )
end

function (iaca::iACA{RowPivType,ColPivType,ConvCritType})(
    rows::AbstractVector{Int}, Ft::AbstractVector{Int}; maxrank::Int=40
) where {RowPivType<:ValuePivStrat,ColPivType<:TreeMimicryPivoting,ConvCritType<:ConvCrit}
    return iaca(rows, Ft, maxrank)
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
    colbuffer::AbstractArray{K},
    rowbuffer::AbstractArray{K},
    maxrank::Int;
    rows=zeros(Int, maxrank),
    cols=zeros(Int, maxrank),
    rowidcs=Vector(1:size(A, 1)),
    colidcs=Vector(1:size(A, 2)),
) where {K,RowPivType<:ValuePivStrat,ColPivType<:MimicryPivoting,ConvCritType<:ConvCrit}
    return iaca(rowidcs, colidcs)(A, colbuffer, rowbuffer, maxrank, rows, cols, rowidcs)
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
    colbuffer::AbstractArray{K},
    rowbuffer::AbstractArray{K},
    maxrank::Int;
    rows=zeros(Int, maxrank),
    cols=zeros(Int, maxrank),
    rowidcs=Vector(1:size(A, 1)),
    colidcs=Vector(1:size(A, 2)),
) where {K,RowPivType<:ValuePivStrat,ColPivType<:TreeMimicryPivoting,ConvCritType<:ConvCrit}
    return iaca(rowidcs, colidcs, maxrank)(
        A, colbuffer, rowbuffer, maxrank, rows, cols, rowidcs
    )
end
=#

function (iaca::iACA{RowPivType,ColPivType,ConvCritType})(
    rowidcs::AbstractVector{Int}, colidcs::AbstractVector{Int}, maxrank::Int
) where {RowPivType<:MaximumValue,ColPivType<:GeoPivStrat,ConvCritType<:ConvCrit}
    colpivstrat = _buildpivstrat(iaca.columnpivoting, rowidcs, colidcs, maxrank)
    return iACA(iaca.rowpivoting(rowidcs), colpivstrat, iaca.convergence(maxrank))
end

function reset!(
    iaca::iACA{RP,CP,CC}, rowidcs::AbstractVector{Int}, colidcs::AbstractVector{Int}
) where {RP<:MaximumValueFunctor,CP<:GeoPivStratFunctor,CC<:ConvCritFunctor}
    reset!(iaca.rowpivoting, rowidcs)
    reset!(iaca.columnpivoting, rowidcs, colidcs)
    reset!(iaca.convergence)
    return nothing
end

function (iaca::iACA{RP,CP,CC})(
    A,
    colbuffer::AbstractArray{K},
    rowbuffer::AbstractArray{K},
    rowpivs::T,
    colpivs::T,
    rowidcs::T,
    colidcs::T,
    maxrank::Int;
) where {
    K,RP<:MaximumValueFunctor,CP<:GeoPivStratFunctor,CC<:ConvCritFunctor,T<:Vector{Int}
}
    reset!(iaca, rowidcs, colidcs)
    return iaca(A, colbuffer, rowbuffer, rowpivs, colpivs, rowidcs, maxrank)
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
    rowpivs::T,
    colpivs::T,
    rowidcs::T,
    maxrank::Int,
) where {
    K,
    RowPivType<:ValuePivStratFunctor,
    ColPivType<:GeoPivStratFunctor,
    ConvCritType<:ConvCritFunctor,
    T<:Vector{Int},
}
    maxrow = length(rowidcs)
    npivot = 1

    colpivs[npivot] = iaca.columnpivoting()
    nextrc!(
        view(colbuffer, 1:maxrow, npivot:npivot),
        A,
        view(rowidcs, 1:maxrow),
        view(colpivs, npivot:npivot),
    )
    normF!(iaca.convergence.estimator, colbuffer[1:maxrow, npivot], npivot)
    rowbuffer[1, 1] = K(1.0)
    rowpivs[npivot] = iaca.rowpivoting(colbuffer[1:maxrow, npivot])

    npivot, conv = iaca.convergence(colbuffer[1:maxrow, npivot], npivot)

    while conv && npivot < maxrank
        npivot += 1

        colpivs[npivot] = iaca.columnpivoting(npivot)

        nextrc!(
            view(colbuffer, 1:maxrow, npivot:npivot),
            A,
            view(rowidcs, 1:maxrow),
            view(colpivs, npivot:npivot),
        )

        # Norm update
        normF!(iaca.convergence.estimator, colbuffer[1:maxrow, npivot], npivot)

        rowbuffer[npivot, npivot] = K(1.0)
        for k in 1:(npivot - 1)
            @views rowbuffer[k, npivot] =
                colbuffer[rowpivs[k], k]^-1 * colbuffer[rowpivs[k], npivot]
            for kk in 1:maxrow
                @views colbuffer[kk, npivot] -= colbuffer[kk, k] * rowbuffer[k, npivot]
            end
        end
        rowpivs[npivot] = iaca.rowpivoting(colbuffer[1:maxrow, npivot])
        npivot, conv = iaca.convergence(colbuffer[1:maxrow, npivot], npivot)
    end

    return npivot, rowidcs[rowpivs[1:npivot]], colpivs[1:npivot]
end
