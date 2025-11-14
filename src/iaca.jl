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

function iACA(tpos::Vector{SVector{D,F}}, spos::Vector{SVector{D,F}}) where {D,F<:Real}
    return iACA(
        MaximumValue(),
        MimicryPivoting(tpos, spos),
        FNormExtrapolator(iFNormEstimator(F(1e-4))),
    )
end

function (iaca::iACA{RowPivType,ColPivType,ConvCritType})(
    rows::AbstractVector{Int}, cols::AbstractVector{Int}
) where {RowPivType<:GeoPivStrat,ColPivType<:ValuePivStrat,ConvCritType<:ConvCrit}
    return iACA(iaca.rowpivoting(cols, rows), iaca.columnpivoting(cols), iaca.convergence())
end

function (iaca::iACA{RowPivType,ColPivType,ConvCritType})(
    Fs::AbstractVector{Int}, cols::AbstractVector{Int}, maxrank::Int
) where {RowPivType<:TreeMimicryPivoting,ColPivType<:ValuePivStrat,ConvCritType<:ConvCrit}
    return iACA(
        iaca.rowpivoting(Fs, cols, maxrank), iaca.columnpivoting(cols), iaca.convergence()
    )
end

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

function (iaca::iACA{RowPivType,ColPivType,ConvCritType})(
    rows::AbstractVector{Int}, cols::AbstractVector{Int}
) where {RowPivType<:ValuePivStrat,ColPivType<:GeoPivStrat,ConvCritType<:ConvCrit}
    return iACA(iaca.rowpivoting(rows), iaca.columnpivoting(rows, cols), iaca.convergence())
end

function (iaca::iACA{RowPivType,ColPivType,ConvCritType})(
    rows::AbstractVector{Int}, Ft::AbstractVector{Int}, maxrank::Int
) where {RowPivType<:ValuePivStrat,ColPivType<:TreeMimicryPivoting,ConvCritType<:ConvCrit}
    return iACA(
        iaca.rowpivoting(rows), iaca.columnpivoting(Ft, rows, maxrank), iaca.convergence()
    )
end

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
