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

    aca = aca(K, irange, jrange)
    npivots, U, V = aca(K, rowBuffer, colBuffer, maxrank; rowidcs=irange, colidcs=jrange)
    return HMatrices.RkMatrix(
        colBuffer[:, 1:npivots], Matrix(transpose(rowBuffer[1:npivots, :]))
    )
end

function Base.resize!(A::HMatrices.VectorOfVectors, m::Int, n::Int)
    A.m = m
    ie = m * n
    if ie > length(A.data)
        resize!(A.data, ie)
    end
    return A.k = n
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
