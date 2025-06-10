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

function iACA(
    pos::Vector{SVector{D,F}};
    rowpivoting=LRF.MaximumValue(),
    columnpivoting=IACAPivoting(pos),
    convergence=IncompleteNormEstimator(F[], F(0.0)),
) where {D,F<:Real}
    return iACA(rowpivoting, columnpivoting, convergence)
end

function init(
    iaca::iACA{RPT,CPT,CCT},
    M::LRF.LazyMatrix{Int,K};
    ref=sum(iaca.rowpivoting.pos[M.σ]) / length(M.σ),
) where {K,RPT<:GeoPivStrat,CPT<:LRF.PivStrat,CCT<:LRF.ConvCrit}
    return iACA(
        iaca.rowpivoting(M.τ; ref=ref), iaca.columnpivoting(M.σ), iaca.convergence(M)
    )
end

function init(
    iaca::iACA{RPT,CPT,CCT},
    M::LRF.LazyMatrix{Int,K};
    ref=sum(iaca.columnpivoting.pos[M.τ]) / length(M.τ),
) where {K,RPT<:LRF.PivStrat,CPT<:GeoPivStrat,CCT<:LRF.ConvCrit}
    return iACA(
        iaca.rowpivoting(M.τ), iaca.columnpivoting(M.σ; ref=ref), iaca.convergence(M)
    )
end

function (iaca::iACA{RowPivType,ColPivType,ConvCritType})(
    M::LRF.LazyMatrix{Int,K},
    rowbuffer::AbstractMatrix{K},
    colbuffer::AbstractMatrix{K},
    maxrank::Int,
    tol::F,
) where {
    F<:Real,K,RowPivType<:GeoPivStrat,ColPivType<:LRF.PivStrat,ConvCritType<:LRF.ConvCrit
}
    maxcolumn = size(M, 2)
    rows = zeros(Int, maxrank)
    cols = zeros(Int, maxrank)
    npivot = 1

    rows[npivot] = iaca.rowpivoting()
    @views M.μ(
        rowbuffer[npivot:npivot, 1:maxcolumn],
        M.τ[rows[npivot]:rows[npivot]],
        M.σ[1:maxcolumn],
    )
    iaca.convergence.normUV = norm(rowbuffer[npivot:npivot, 1:maxcolumn])
    colbuffer[1, 1] = K(1.0)
    cols[npivot] = iaca.columnpivoting(rowbuffer[npivot, 1:maxcolumn])

    conv = iaca.convergence(rowbuffer[npivot, 1:maxcolumn], npivot, tol)

    while conv && npivot < maxrank
        npivot += 1

        rows[npivot] = iaca.rowpivoting(npivot)
        @views M.μ(
            rowbuffer[npivot:npivot, 1:maxcolumn],
            M.τ[rows[npivot]:rows[npivot]],
            M.σ[1:maxcolumn],
        )

        # Norm update
        updatenorm!(iaca.convergence, rowbuffer[npivot, 1:maxcolumn], npivot)

        colbuffer[npivot, npivot] = K(1.0)
        for k in 1:(npivot-1)
            @views colbuffer[npivot, k] =
                rowbuffer[k, cols[k]]^-1 * rowbuffer[npivot, cols[k]]
            for kk in 1:maxcolumn
                @views rowbuffer[npivot, kk] -= rowbuffer[k, kk] * colbuffer[npivot, k]
            end
        end

        cols[npivot] = iaca.columnpivoting(rowbuffer[npivot, 1:maxcolumn])
        conv = iaca.convergence(rowbuffer[npivot, 1:maxcolumn], npivot, tol)
    end

    return rows[1:npivot], cols[1:npivot]
end

function (iaca::iACA{RowPivType,ColPivType,ConvCritType})(
    M::LRF.LazyMatrix{Int,K},
    rowbuffer::AbstractMatrix{K},
    colbuffer::AbstractMatrix{K},
    maxrank::Int,
    tol::F;
    verbose=false,
) where {
    F<:Real,K,RowPivType<:LRF.PivStrat,ColPivType<:GeoPivStrat,ConvCritType<:LRF.ConvCrit
}
    maxrow = size(M, 1)
    rows = zeros(Int, maxrank)
    cols = zeros(Int, maxrank)
    npivot = 1

    ptime = @elapsed cols[npivot] = iaca.columnpivoting()
    @views M.μ(
        colbuffer[1:maxrow, npivot:npivot], M.τ[1:maxrow], M.σ[cols[npivot]:cols[npivot]]
    )
    iaca.convergence.normUV = norm(colbuffer[1:maxrow, npivot:npivot])
    rowbuffer[1, 1] = K(1.0)
    rows[npivot] = iaca.rowpivoting(colbuffer[1:maxrow, npivot])

    conv = iaca.convergence(colbuffer[1:maxrow, npivot], npivot, tol)

    while conv && npivot < maxrank
        npivot += 1

        ptime += @elapsed cols[npivot] = iaca.columnpivoting(npivot)
        @views M.μ(
            colbuffer[1:maxrow, npivot:npivot],
            M.τ[1:maxrow],
            M.σ[cols[npivot]:cols[npivot]],
        )

        # Norm update
        updatenorm!(iaca.convergence, colbuffer[1:maxrow, npivot], npivot)

        rowbuffer[npivot, npivot] = K(1.0)
        for k in 1:(npivot-1)
            @views rowbuffer[k, npivot] =
                colbuffer[rows[k], k] .^ -1 * colbuffer[rows[k], npivot]
            for kk in 1:maxrow
                @views colbuffer[kk, npivot] -= colbuffer[kk, k] * rowbuffer[k, npivot]
            end
        end

        rows[npivot] = iaca.rowpivoting(colbuffer[1:maxrow, npivot])
        conv = iaca.convergence(colbuffer[1:maxrow, npivot], npivot, tol)
    end
    if verbose
        return ptime, rows[1:npivot], cols[1:npivot]
    else
        return rows[1:npivot], cols[1:npivot]
    end
end

# ButtomUpCompressor

function (iaca::iACA{RowPivType,ColPivType,ConvCritType})(
    farassembler::Function,
    rowbuffer::AbstractMatrix{K},
    colbuffer::AbstractMatrix{K},
    colidcs::Vector{Int},
    maxrank::Int,
    tol::F,
    ref::SVector{3,F},
    fars::Vector{Int};
    convcrit=IncompleteNormEstimator(F[], F(0.0)),
) where {
    F<:Real,K,RowPivType<:IACAPivoting2,ColPivType<:LRF.PivStrat,ConvCritType<:LRF.ConvCrit
}
    colpivoting = LRF.MaximumValue(zeros(Bool, length(colidcs)))
    maxcolumn = length(colidcs)
    rows = zeros(Int, maxrank)
    cols = zeros(Int, maxrank)
    npivot = 1
    cts = [iaca.rowpivoting.tree.nodes[node].node.data.ct for node in fars]

    rows[npivot], data = iaca.rowpivoting(ref, fars, cts)
    @views farassembler(rowbuffer[npivot:npivot, 1:maxcolumn], [rows[npivot]], colidcs)
    convcrit.normUV = norm(rowbuffer[npivot:npivot, 1:maxcolumn])
    colbuffer[1, 1] = K(1.0)
    cols[npivot] = colpivoting(rowbuffer[npivot, 1:maxcolumn])

    conv = convcrit(rowbuffer[npivot, 1:maxcolumn], npivot, tol)

    while conv && npivot < maxrank
        npivot += 1

        rows[npivot], data = iaca.rowpivoting(ref, fars, cts, rows[1:(npivot-1)], data)
        @views farassembler(rowbuffer[npivot:npivot, 1:maxcolumn], [rows[npivot]], colidcs)

        # Norm update
        updatenorm!(convcrit, rowbuffer[npivot, 1:maxcolumn], npivot)

        colbuffer[npivot, npivot] = K(1.0)
        for k in 1:(npivot-1)
            @views colbuffer[npivot, k] =
                rowbuffer[k, cols[k]]^-1 * rowbuffer[npivot, cols[k]]
            for kk in 1:maxcolumn
                @views rowbuffer[npivot, kk] -= rowbuffer[k, kk] * colbuffer[npivot, k]
            end
        end

        cols[npivot] = colpivoting(rowbuffer[npivot, 1:maxcolumn])
        if isapprox(norm(rowbuffer[npivot, 1:maxcolumn]), 0.0; atol=eps(real(K)))
            conv = false
            npivot -= 1
        else
            conv = convcrit(rowbuffer[npivot, 1:maxcolumn], npivot, tol)
        end
    end

    return rows[1:npivot], colidcs[cols[1:npivot]]
end

function (iaca::iACA{RowPivType,ColPivType,ConvCritType})(
    farassembler::Function,
    rowbuffer::AbstractMatrix{K},
    colbuffer::AbstractMatrix{K},
    rowidcs::Vector{Int},
    maxrank::Int,
    tol::F,
    ref::SVector{3,F},
    fars::Vector{Int};
    convcrit=IncompleteNormEstimator(F[], F(0.0)),
    verbose=false,
) where {
    F<:Real,K,RowPivType<:LRF.PivStrat,ColPivType<:IACAPivoting2,ConvCritType<:LRF.ConvCrit
}
    rowpivoting = LRF.MaximumValue(zeros(Bool, length(rowidcs)))
    maxrow = length(rowidcs)
    rows = zeros(Int, maxrank)
    cols = zeros(Int, maxrank)
    npivot = 1
    cts = [iaca.columnpivoting.tree.nodes[node].node.data.ct for node in fars]

    ptime = @elapsed cols[npivot], data = iaca.columnpivoting(ref, fars, cts)
    @views farassembler(colbuffer[1:maxrow, npivot:npivot], rowidcs, [cols[npivot]])
    convcrit.normUV = norm(colbuffer[1:maxrow, npivot:npivot])
    rowbuffer[1, 1] = K(1.0)
    rows[npivot] = rowpivoting(colbuffer[1:maxrow, npivot])

    conv = convcrit(colbuffer[1:maxrow, npivot], npivot, tol)

    while conv && npivot < maxrank
        npivot += 1

        ptime += @elapsed cols[npivot], data = iaca.columnpivoting(
            ref, fars, cts, cols[1:(npivot-1)], data
        )
        @views farassembler(colbuffer[1:maxrow, npivot:npivot], rowidcs, [cols[npivot]])

        # Norm update
        updatenorm!(convcrit, colbuffer[1:maxrow, npivot], npivot)

        rowbuffer[npivot, npivot] = K(1.0)
        for k in 1:(npivot-1)
            @views rowbuffer[k, npivot] =
                colbuffer[rows[k], k] .^ -1 * colbuffer[rows[k], npivot]
            for kk in 1:maxrow
                @views colbuffer[kk, npivot] -= colbuffer[kk, k] * rowbuffer[k, npivot]
            end
        end

        rows[npivot] = rowpivoting(colbuffer[1:maxrow, npivot])
        if isapprox(norm(colbuffer[1:maxrow, npivot]), 0.0; atol=eps(real(K)))
            conv = false
            npivot -= 1
        else
            conv = convcrit(colbuffer[1:maxrow, npivot], npivot, tol)
        end
    end

    if verbose
        return ptime, rowidcs[rows[1:npivot]], cols[1:npivot]
    else
        return rowidcs[rows[1:npivot]], cols[1:npivot]
    end
end
