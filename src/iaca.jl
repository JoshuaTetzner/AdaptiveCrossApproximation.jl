"""
    iACA{RowPivType,ColPivType,ConvCritType}

Incomplete Adaptive Cross Approximation (iACA) compressor.

Unlike standard ACA, iACA computes only one side per iteration and relies on geometric
pivoting strategies (for example mimicry or tree mimicry) to select pivots from spatial
information. This reduces matrix entry evaluations in hierarchical matrix construction,
where only selected row or column samples are required.

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

"""
    iACA(tpos, spos)

Create a default incomplete ACA compressor for geometrically indexed row/column sets.

# Arguments
- `tpos::Vector{SVector{D,F}}`: geometric positions for test indices
- `spos::Vector{SVector{D,F}}`: geometric positions for trial indices

# Returns
An `iACA` instance using `MaximumValue`/`MimicryPivoting` with `FNormExtrapolator`.

# See also
`iACA`, `MimicryPivoting`, `FNormExtrapolator`
"""
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
