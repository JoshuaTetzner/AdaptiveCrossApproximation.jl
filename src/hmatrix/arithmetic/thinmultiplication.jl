# Thin products of a hierarchical block with dense panels in tree order:
# Y += alpha*op(H)*X (left) or Y += alpha*X*op(H) (right). Each terminal block
# applies the whole panel with BLAS-3 calls; low-rank blocks borrow
# `rank * panel width` entries of `buffer`.

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

@inline _thinpanels(Y, X, rows, cols, ::Val{:left}) = (view(Y, rows, :), view(X, cols, :))
@inline _thinpanels(Y, X, rows, cols, ::Val{:right}) = (view(Y, :, cols), view(X, :, rows))

function _thinexecute!(Y, X, H, alpha, buffer, operation::Val{O}, side::Val{S}) where {O,S}
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
            U, V = view(R.U, localrows, :), view(R.V, :, localcols)
            if O === :N
                _thinlowrank!(target, source, U, V, alpha, buffer, side)
            else
                _thinlowrank!(
                    target,
                    source,
                    _orient(V, operation),
                    _orient(U, operation),
                    alpha,
                    buffer,
                    side,
                )
            end
        else
            D = _orient(view(W.near[csr.storageidx[k]], localrows, localcols), operation)
            if S === :left
                mul!(target, D, source, alpha, one(alpha))
            else
                mul!(target, source, D, alpha, one(alpha))
            end
        end
    end
    return Y
end

"""
    _thinmul!(Y, H::HMatrixBlock, X, α, β, scratch, Val(side))

Product of a hierarchical block with a dense panel: `Y = α*H*X + β*Y`
(`side=:left`) or `Y = α*X*H + β*Y` (`side=:right`), in tree order. Every terminal
block applies the whole panel at once, a low-rank block as `U*(V*X)`; the cost
is the storage of `H` times the panel width. This is the kernel behind the
products of low-rank terminals with hierarchical panels and the triangular
solves.
"""
function _thinmul!(
    Y::AbstractMatrix{T}, H, X, alpha, beta, scratch, side::Val{S}
) where {T,S}
    m, n = size(H)
    if S === :left
        size(X, 1) == n && size(Y) == (m, size(X, 2)) ||
            throw(DimensionMismatch("incompatible hierarchical left product dimensions"))
    else
        size(X, 2) == m && size(Y) == (size(X, 1), n) ||
            throw(DimensionMismatch("incompatible hierarchical right product dimensions"))
    end
    alpha, beta = convert(T, alpha), convert(T, beta)
    if iszero(beta)
        fill!(Y, zero(T))
    elseif !isone(beta)
        rmul!(Y, beta)
    end
    (iszero(alpha) || isempty(Y) || isempty(X)) && return Y
    width = S === :left ? size(X, 2) : size(X, 1)
    _growscratch!(scratch.small, checked_mul(_maxrank(H.workspace, H.entries), width))
    return _thinexecute!(Y, X, H, alpha, scratch.small, Val(:N), side)
end

# Y = alpha*H*X + beta*Y and Y = alpha*X*H + beta*Y for a hierarchical block H.
_thinleft!(Y, H::HMatrixBlock, X, alpha, beta, scratch) =
    _thinmul!(Y, H, X, alpha, beta, scratch, Val(:left))
_thinright!(Y, X, H::HMatrixBlock, alpha, beta, scratch) =
    _thinmul!(Y, H, X, alpha, beta, scratch, Val(:right))
