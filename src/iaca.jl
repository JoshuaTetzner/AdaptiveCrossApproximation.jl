"""
    IACA{RowPivType,ColPivType,ConvCritType}

Incomplete Adaptive Cross Approximation (IACA) compressor.

Unlike standard ACA, IACA computes only one side per iteration and relies on geometric
pivoting strategies (for example mimicry or tree mimicry) to select pivots from spatial
information. This reduces matrix entry evaluations in hierarchical matrix construction,
where only selected row or column samples are required. One pivoting direction must be
geometric and the other value-based; the algorithm returns the selected row and column
indices rather than a materialized low-rank factorization.

# Fields

  - `rowpivoting::RowPivType`: Strategy for selecting row pivots
  - `columnpivoting::ColPivType`: Strategy for selecting column pivots
  - `convergence::ConvCritType`: Convergence criterion
"""
struct IACA{RowPivType,ColPivType,ConvCritType}
    rowpivoting::RowPivType
    columnpivoting::ColPivType
    convergence::ConvCritType

    function IACA(rowpivoting, columnpivoting, convergence)
        return new{typeof(rowpivoting),typeof(columnpivoting),typeof(convergence)}(
            rowpivoting, columnpivoting, convergence
        )
    end
end

@inline _iaca_normF!(convcrit::ConvCritFunctor, rcbuffer, npivot) =
    normF!(convcrit.estimator, rcbuffer, npivot)

"""
    IACA(tpos, spos)

Create a default incomplete ACA compressor for geometrically indexed row/column sets.

# Arguments

  - `tpos::Vector{SVector{D,F}}`: geometric positions for test indices
  - `spos::Vector{SVector{D,F}}`: geometric positions for trial indices

# Returns

An `IACA` instance using `MaximumValue` for rows, `MimicryPivoting` for columns,
and `FNormExtrapolator` for convergence.

# See also

`IACA`, `MimicryPivoting`, `FNormExtrapolator`
"""
function IACA(tpos::Vector{SVector{D,F}}, spos::Vector{SVector{D,F}}) where {D,F<:Real}
    return IACA(
        MaximumValue(),
        MimicryPivoting(tpos, spos),
        FNormExtrapolator(FNormEstimator(F(1e-4))),
    )
end

function (iaca::IACA{RowPivType,ColPivType,ConvCritType})(
    rowidcs::AbstractVector{Int}, colidcs::AbstractVector{Int}, maxrank::Int
) where {RowPivType<:GeoPivStrat,ColPivType<:MaximumValue,ConvCritType<:ConvCrit}
    rowpivstrat = _buildpivstrat(iaca.rowpivoting, colidcs, rowidcs, maxrank)
    return IACA(rowpivstrat, iaca.columnpivoting(colidcs), iaca.convergence(maxrank))
end

function (iaca::IACA{RowPivType,ColPivType,ConvCritType})(
    rowidcs::AbstractVector{Int}, colidcs::AbstractVector{Int}, maxrank::Int
) where {
    RowPivType<:TreeMimicryPivoting{<:Any,<:Any,<:Any,<:EFIEDirectionalFilter},
    ColPivType<:MaximumValue,
    ConvCritType<:ConvCrit,
}
    convcrit = iaca.convergence(maxrank)
    rowpivstrat = _buildpivstrat(convcrit, iaca.rowpivoting, colidcs, rowidcs, maxrank)
    return IACA(rowpivstrat, iaca.columnpivoting(colidcs), convcrit)
end

function reset!(
    iaca::IACA{RP,CP,CC}, rowidcs::AbstractVector{Int}, colidcs::AbstractVector{Int}
) where {RP<:GeoPivStratFunctor,CP<:MaximumValueFunctor,CC<:ConvCritFunctor}
    reset!(iaca.rowpivoting, colidcs, rowidcs)
    reset!(iaca.columnpivoting, colidcs)
    reset!(iaca.convergence)
    return nothing
end

function (iaca::IACA{RP,CP,CC})(
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
    iaca(A, colbuffer, rowbuffer, rowpivots, colpivots, colidcs, maxrank)

Low-level IACA iteration with geometric row pivoting and value-based column
pivoting. This method operates on a stateful compressor built for a particular
index block.

# Arguments

  - `A`: Matrix to compress
  - `colbuffer::AbstractMatrix{K}`: Buffer for column data
  - `rowbuffer::AbstractMatrix{K}`: Buffer for row data
  - `rowpivots::Vector{Int}`: Storage for selected row indices
  - `colpivots::Vector{Int}`: Storage for selected column indices
  - `colidcs::Vector{Int}`: Column index range
  - `maxrank::Int`: Maximum number of pivots

# Returns

  - `npivot::Int`: Number of pivots computed
  - `rows::Vector{Int}`: Selected row indices
  - `cols::Vector{Int}`: Selected column indices (global)
"""
function (iaca::IACA{RowPivType,ColPivType,ConvCritType})(
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
        rowpivs[npivot],
        view(colidcs, 1:maxcolumn),
    )
    _iaca_normF!(iaca.convergence, view(rowbuffer, npivot, 1:maxcolumn), npivot)
    colbuffer[1, 1] = K(1.0)
    colpivs[npivot] = iaca.columnpivoting(view(rowbuffer, npivot, 1:maxcolumn))

    npivot, conv = iaca.convergence(view(rowbuffer, npivot, 1:maxcolumn), npivot)

    while conv && npivot < maxrank
        npivot += 1

        rowpivs[npivot] = iaca.rowpivoting(npivot)
        nextrc!(
            view(rowbuffer, npivot:npivot, 1:maxcolumn),
            A,
            rowpivs[npivot],
            view(colidcs, 1:maxcolumn),
        )

        # Norm update
        _iaca_normF!(iaca.convergence, view(rowbuffer, npivot, 1:maxcolumn), npivot)

        colbuffer[npivot, npivot] = K(1.0)
        for k in 1:(npivot - 1)
            @views colbuffer[npivot, k] =
                rowbuffer[k, colpivs[k]]^-1 * rowbuffer[npivot, colpivs[k]]
            for kk in 1:maxcolumn
                @views rowbuffer[npivot, kk] -= rowbuffer[k, kk] * colbuffer[npivot, k]
            end
        end
        colpivs[npivot] = iaca.columnpivoting(view(rowbuffer, npivot, 1:maxcolumn))
        npivot, conv = iaca.convergence(view(rowbuffer, npivot, 1:maxcolumn), npivot)
    end

    return npivot, rowpivs[1:npivot], colidcs[colpivs[1:npivot]]
end

function (iaca::IACA{RowPivType,ColPivType,ConvCritType})(
    rowidcs::AbstractVector{Int}, colidcs::AbstractVector{Int}, maxrank::Int
) where {RowPivType<:MaximumValue,ColPivType<:GeoPivStrat,ConvCritType<:ConvCrit}
    colpivstrat = _buildpivstrat(iaca.columnpivoting, rowidcs, colidcs, maxrank)
    return IACA(iaca.rowpivoting(rowidcs), colpivstrat, iaca.convergence(maxrank))
end

function (iaca::IACA{RowPivType,ColPivType,ConvCritType})(
    rowidcs::AbstractVector{Int}, colidcs::AbstractVector{Int}, maxrank::Int
) where {
    RowPivType<:MaximumValue,
    ColPivType<:TreeMimicryPivoting{<:Any,<:Any,<:Any,<:EFIEDirectionalFilter},
    ConvCritType<:ConvCrit,
}
    convcrit = iaca.convergence(maxrank)
    colpivstrat = _buildpivstrat(convcrit, iaca.columnpivoting, rowidcs, colidcs, maxrank)
    return IACA(iaca.rowpivoting(rowidcs), colpivstrat, convcrit)
end

function reset!(
    iaca::IACA{RP,CP,CC}, rowidcs::AbstractVector{Int}, colidcs::AbstractVector{Int}
) where {RP<:MaximumValueFunctor,CP<:GeoPivStratFunctor,CC<:ConvCritFunctor}
    reset!(iaca.rowpivoting, rowidcs)
    reset!(iaca.columnpivoting, rowidcs, colidcs)
    reset!(iaca.convergence)
    return nothing
end

function (iaca::IACA{RP,CP,CC})(
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
    iaca(A, colbuffer, rowbuffer, rowpivots, colpivots, rowidcs, maxrank)

Low-level IACA iteration with value-based row pivoting and geometric column
pivoting. This method operates on a stateful compressor built for a particular
index block.

# Arguments

  - `A`: Matrix to compress
  - `colbuffer::AbstractArray{K}`: Buffer for column data
  - `rowbuffer::AbstractArray{K}`: Buffer for row data
  - `rowpivots::Vector{Int}`: Storage for selected row indices
  - `colpivots::Vector{Int}`: Storage for selected column indices
  - `rowidcs::Vector{Int}`: Row index range
  - `maxrank::Int`: Maximum number of pivots

# Returns

  - `npivot::Int`: Number of pivots computed
  - `rows::Vector{Int}`: Selected row indices (global)
  - `cols::Vector{Int}`: Selected column indices
"""
function (iaca::IACA{RowPivType,ColPivType,ConvCritType})(
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
        colpivs[npivot],
    )
    _iaca_normF!(iaca.convergence, view(colbuffer, 1:maxrow, npivot), npivot)
    rowbuffer[1, 1] = K(1.0)
    rowpivs[npivot] = iaca.rowpivoting(view(colbuffer, 1:maxrow, npivot))

    npivot, conv = iaca.convergence(view(colbuffer, 1:maxrow, npivot), npivot)

    while conv && npivot < maxrank
        npivot += 1

        colpivs[npivot] = iaca.columnpivoting(npivot)

        nextrc!(
            view(colbuffer, 1:maxrow, npivot:npivot),
            A,
            view(rowidcs, 1:maxrow),
            colpivs[npivot],
        )

        # Norm update
        _iaca_normF!(iaca.convergence, view(colbuffer, 1:maxrow, npivot), npivot)

        rowbuffer[npivot, npivot] = K(1.0)
        for k in 1:(npivot - 1)
            @views rowbuffer[k, npivot] =
                colbuffer[rowpivs[k], k]^-1 * colbuffer[rowpivs[k], npivot]
            for kk in 1:maxrow
                @views colbuffer[kk, npivot] -= colbuffer[kk, k] * rowbuffer[k, npivot]
            end
        end
        rowpivs[npivot] = iaca.rowpivoting(view(colbuffer, 1:maxrow, npivot))
        npivot, conv = iaca.convergence(view(colbuffer, 1:maxrow, npivot), npivot)
    end

    return npivot, rowidcs[rowpivs[1:npivot]], colpivs[1:npivot]
end
