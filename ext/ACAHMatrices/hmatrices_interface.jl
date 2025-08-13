function (aca::ACA)(
    K::HMatrices.PermutedMatrix{TT,T},
    rowtree::ClusterTree,
    coltree::ClusterTree,
    bufs::HMatrices.ACABuffer,
) where {TT<:HMatrices.KernelMatrix,T}
    irange = HMatrices.index_range(rowtree)
    jrange = HMatrices.index_range(coltree)
    maxrank = min(length(irange), length(jrange))

    resize!(bufs.A, maxrank, length(jrange))
    resize!(bufs.B, length(irange), maxrank)
    # The ACA expects an AbstractMatrix whlie HMatrices.VectorOfVectors is not its subtype, 
    # hence the data Vector is reshaped here and directly used
    rowBuffer = reshape(
        bufs.A.data[1:(maxrank * length(jrange))], (maxrank, length(jrange))
    )
    colBuffer = reshape(
        bufs.B.data[1:(length(irange) * maxrank)], (length(irange), maxrank)
    )
    update_conv_idcs!(aca.convergence, K, irange, jrange)
    cur_aca = AdaptiveCrossApproximation.ACA(;
        rowpivoting=update_pivot_idcs(K, aca.rowpivoting, irange, jrange, true),
        columnpivoting=update_pivot_idcs(K, aca.columnpivoting, irange, jrange, false),
        convergence=aca.convergence,
    )

    npivots, U, V = cur_aca(
        K, rowBuffer, colBuffer, maxrank; rowidcs=irange, colidcs=jrange
    )
    return HMatrices.RkMatrix(
        colBuffer[:, 1:npivots], Matrix(transpose(rowBuffer[1:npivots, :]))
    )
end

function update_pivot_idcs(
    _::HMatrices.PermutedMatrix{TT,T},
    pivoting::AdaptiveCrossApproximation.ValuePivStrat,
    rowidcs::UnitRange{Int},
    colidcs::UnitRange{Int},
    is_row::Bool,
) where {TT<:HMatrices.KernelMatrix,T}
    len = is_row ? length(rowidcs) : length(colidcs)
    return pivoting(len)
end

function update_pivot_idcs(
    _::HMatrices.PermutedMatrix{TT,T},
    pivoting::AdaptiveCrossApproximation.GeoPivStrat,
    rowidcs::UnitRange{Int},
    colidcs::UnitRange{Int},
    is_row::Bool,
) where {TT<:HMatrices.KernelMatrix,T}
    idcsvec = is_row ? Vector(rowidcs) : Vector(colidcs)
    return pivoting(idcsvec)
end

function update_pivot_idcs(
    K::HMatrices.PermutedMatrix{TT,T},
    pivoting::AdaptiveCrossApproximation.ConvPivStrat,
    rowidcs::UnitRange{Int},
    colidcs::UnitRange{Int},
    _::Bool,
) where {TT<:HMatrices.KernelMatrix,T}
    update_conv_idcs!(pivoting.convcrit, K, rowidcs, colidcs)
    return typeof(pivoting)(pivoting.convcrit, pivoting.rc)
end

function update_pivot_idcs(
    K::HMatrices.PermutedMatrix{TT,T},
    pivoting::AdaptiveCrossApproximation.CombinedPivStrat,
    rowidcs::UnitRange{Int},
    colidcs::UnitRange{Int},
    is_row::Bool,
) where {TT<:HMatrices.KernelMatrix,T}
    curr_strats = Vector{AdaptiveCrossApproximation.PivStrat}(
        undef, length(pivoting.strats)
    )
    for (i, strat) in enumerate(pivoting.strats)
        curr_strats[i] = update_pivot_idcs(K, strat, rowidcs, colidcs, is_row)
    end
    update_conv_idcs!(pivoting.convcrit, K, rowidcs, colidcs)
    return AdaptiveCrossApproximation.CombinedPivStrat(pivoting.convcrit, curr_strats)
end

function Base.resize!(A::HMatrices.VectorOfVectors, m::Int, n::Int)
    A.m = m
    ie = m * n
    if ie > length(A.data)
        resize!(A.data, ie)
    end
    return A.k = n
end

function update_conv_idcs!(
    convcrit::AdaptiveCrossApproximation.CombinedConvCrit,
    K::HMatrices.PermutedMatrix{TT,T},
    irange::UnitRange{Int},
    jrange::UnitRange{Int},
) where {TT<:HMatrices.KernelMatrix,T}
    for crit in convcrit.crits
        update_conv_idcs!(crit, K, irange, jrange)
    end
end

update_conv_idcs!(
    convcrit::AdaptiveCrossApproximation.ConvCrit,
    K::HMatrices.PermutedMatrix{TT,T},
    irange::UnitRange{Int},
    jrange::UnitRange{Int},
) where {TT<:HMatrices.KernelMatrix,T} = nothing

function update_conv_idcs!(
    convcrit::AdaptiveCrossApproximation.RandomSampling{F,G},
    K::HMatrices.PermutedMatrix{TT,T},
    irange::UnitRange{Int},
    jrange::UnitRange{Int},
) where {F<:Real,G,TT<:HMatrices.KernelMatrix,T}
    convcrit.indices = hcat(
        rand(1:length(irange), convcrit.nsamples), rand(1:length(jrange), convcrit.nsamples)
    )
    return convcrit.rest = [K.data[rc[1], rc[2]][1] for rc in eachrow(convcrit.indices)]
end

Base.size(K::HMatrices.VectorOfVectors) = K.m, K.k

function AdaptiveCrossApproximation.nextrc!(
    buf, A::HMatrices.PermutedMatrix{TT,T}, i, j
) where {TT<:HMatrices.KernelMatrix,T}
    """
    overloaded to use A.data.f ,i.e. blkassembler, instead of Base.getindex
    """
    permuted_irange = A.rowperm[Vector(i)]
    permuted_jrange = A.colperm[Vector(j)]
    return A.data.f(buf, permuted_irange, permuted_jrange)
end
