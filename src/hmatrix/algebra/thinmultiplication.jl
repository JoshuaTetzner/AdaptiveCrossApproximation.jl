# Thin products of a hierarchical block with dense panels in tree order:
# Y = alpha*op(H)*X + beta*Y (left) or Y = alpha*X*op(H) + beta*Y (right). Each
# terminal block applies the whole panel with BLAS-3 calls; low-rank blocks need
# `maximum_rank * panel_width` entries of `scratch.small`.

function _checkthinblocks(Y, X, H, scratch)
    W, csr = H.workspace, H.workspace.csr
    maximumrank = 0
    for k in H.entries
        _checkworkspaceblock(W, k)
        block = csr.isfar[k] ? W.far[csr.storageidx[k]] : W.near[csr.storageidx[k]]
        _checkproductalias(Y, block)
        for buffer in (scratch.left, scratch.right, scratch.small)
            _checkproductalias(buffer, block)
        end
        if csr.isfar[k]
            maximumrank = max(maximumrank, rank(W.far[csr.storageidx[k]]))
        end
    end
    return maximumrank
end

function _thinlowrank!(Y, X, U, V, alpha, buffer, ::Val{:left})
    r, q = size(U, 2), size(X, 2)
    r == 0 && return nothing
    K = reshape(view(buffer, 1:(r * q)), r, q)
    mul!(K, V, X)
    mul!(Y, U, K, alpha, one(alpha))
    return nothing
end

function _thinlowrank!(Y, X, U, V, alpha, buffer, ::Val{:right})
    r, q = size(U, 2), size(X, 1)
    r == 0 && return nothing
    K = reshape(view(buffer, 1:(q * r)), q, r)
    mul!(K, X, U)
    mul!(Y, K, V, alpha, one(alpha))
    return nothing
end

_thindense!(Y, X, D, alpha, ::Val{:left}) = mul!(Y, D, X, alpha, one(alpha))
_thindense!(Y, X, D, alpha, ::Val{:right}) = mul!(Y, X, D, alpha, one(alpha))

@inline _thinpanels(Y, X, rows, cols, ::Val{:left}) = (view(Y, rows, :), view(X, cols, :))
@inline _thinpanels(Y, X, rows, cols, ::Val{:right}) = (view(Y, :, cols), view(X, :, rows))

function _thinfarblock!(
    Y, X, R, rows, cols, alpha, buffer, operation::Val{O}, side
) where {O}
    U, V = view(R.U, rows, :), view(R.V, :, cols)
    if O === :N
        _thinlowrank!(Y, X, U, V, alpha, buffer, side)
    else
        _thinlowrank!(
            Y, X, _orient(V, operation), _orient(U, operation), alpha, buffer, side
        )
    end
    return nothing
end

function _thinexecute!(Y, X, H, alpha, buffer, operation::Val{O}, side) where {O}
    W, csr = H.workspace, H.workspace.csr
    for k in H.entries
        rr, cc = W.rowranges[W.entryrows[k]], W.colranges[csr.colnodes[k]]
        rows, cols = _overlap(rr, H.rows), _overlap(cc, H.cols)
        localrows, localcols = _relative(rows, rr), _relative(cols, cc)
        outputrows, outputcols = _relative(rows, H.rows), _relative(cols, H.cols)
        if O !== :N
            outputrows, outputcols = outputcols, outputrows
        end
        target, source = _thinpanels(Y, X, outputrows, outputcols, side)
        if csr.isfar[k]
            R = W.far[csr.storageidx[k]]
            _thinfarblock!(
                target, source, R, localrows, localcols, alpha, buffer, operation, side
            )
        else
            D = W.near[csr.storageidx[k]]
            _thindense!(
                target,
                source,
                _orient(view(D, localrows, localcols), operation),
                alpha,
                side,
            )
        end
    end
    return Y
end

function _thinmul!(
    Y::AbstractMatrix{T}, H, X, alpha, beta, scratch, side::Val{S}
) where {T,S}
    Base.require_one_based_indexing(Y, X)
    dims = size(H)
    if S === :left
        size(X, 1) == dims[2] && size(Y) == (dims[1], size(X, 2)) ||
            throw(DimensionMismatch("incompatible hierarchical left product dimensions"))
        q = size(X, 2)
    else
        size(X, 2) == dims[1] && size(Y) == (size(X, 1), dims[2]) ||
            throw(DimensionMismatch("incompatible hierarchical right product dimensions"))
        q = size(X, 1)
    end
    alpha, beta = convert(T, alpha), convert(T, beta)
    isfinite(alpha) && isfinite(beta) ||
        throw(ArgumentError("alpha and beta must be finite"))
    _checkproductalias(Y, X)
    for buffer in (scratch.left, scratch.right, scratch.small)
        _checkproductalias(buffer, Y)
        _checkproductalias(buffer, X)
    end
    maximumrank = _checkthinblocks(Y, X, H, scratch)
    smallsize = Base.Checked.checked_mul(maximumrank, q)
    if !iszero(alpha) && !isempty(Y) && !isempty(X)
        _growscratch!(scratch.small, smallsize)
    end
    if iszero(beta)
        fill!(Y, zero(T))
    elseif !isone(beta)
        rmul!(Y, beta)
    end
    (iszero(alpha) || isempty(Y) || isempty(X)) && return Y
    _thinexecute!(Y, X, H, alpha, scratch.small, Val(:N), side)
    return Y
end

# Y = alpha*H*X + beta*Y and Y = alpha*X*H + beta*Y for a hierarchical block H.
_thinleft!(Y, H::HMatrixBlock, X, alpha, beta, scratch) =
    _thinmul!(Y, H, X, alpha, beta, scratch, Val(:left))
_thinright!(Y, X, H::HMatrixBlock, alpha, beta, scratch) =
    _thinmul!(Y, H, X, alpha, beta, scratch, Val(:right))
