"""
    _SolvePlan(H::HMatrixTriangular, B::HMatrixBlock; side=:left, batchrank=128)

Reusable plan of the triangular solve `H*X = B` (`side=:left`) or `X*H = B`
(`side=:right`) with a hierarchical right-hand side `B`, overwritten by `X`. The
LU factorization uses it for the off-diagonal panels `U12 = L11 \\ A12` and
`L21 = A21 / U11`.

Each step owns one terminal block of `B`, in the elimination order of the
factor: it first subtracts the product of the coefficient block of `H` with the
already solved blocks before it (a [`_ProductPlan`](@ref)), then solves with the
factor of the diagonal cluster of its rows (left) or columns (right). A low-rank
block is solved through its left factor `U` (left) or right factor `V` (right)
alone, so the solve itself does not change its rank. Steps are executed by
[`_executesolveoperation!`](@ref). Ranks may change between executions, the
partition may not.
"""
struct _SolvePlan{T,H,B,S,P}
    factor::H
    rhs::B
    subfactors::Vector{H}
    entries::Vector{Int}
    submap::Vector{Int}
    updatemap::Vector{Int}
    products::Vector{P}
    side::S
end

function _solveblock(W, rows, cols)
    block = blockview(W, rows, cols)
    entries = block.entries isa Vector{Int} ? block.entries : collect(block.entries)
    return HMatrixBlock(W, rows, cols, entries)
end

function _SolvePlan(
    H::HMatrixTriangular{T}, B::HMatrixBlock{T}; side::Symbol=:left, batchrank::Integer=128
) where {T<:LinearAlgebra.BlasFloat}
    side in (:left, :right) || throw(ArgumentError("side must be :left or :right"))
    return if side === :left
        _solveplan(H, B, Val(:left); batchrank)
    else
        _solveplan(H, B, Val(:right); batchrank)
    end
end

function _solveplan(
    H::HMatrixTriangular{T}, B::HMatrixBlock{T}, side::Val{S}; batchrank::Integer=128
) where {T<:LinearAlgebra.BlasFloat,S}
    batchrank > 0 || throw(ArgumentError("batchrank must be positive"))
    W, csr = B.workspace, B.workspace.csr
    axis = S === :left ? B.rows : B.cols
    tree = S === :left ? testtree(W.tree) : trialtree(W.tree)
    perm = S === :left ? W.rowperm : W.colperm
    tree === testtree(H.workspace.tree) ||
        throw(ArgumentError("RHS solve axis requires the factor's cluster-tree object"))
    perm == H.workspace.rowperm || throw(ArgumentError("RHS solve axis ordering differs"))
    axis == H.range || throw(DimensionMismatch("RHS solve range differs from factor"))
    for k in B.entries
        rr, cc = _productranges(W, k)
        _overlap(rr, B.rows) == rr && _overlap(cc, B.cols) == cc || throw(
            ArgumentError("hierarchical RHS views must contain complete physical blocks"),
        )
    end
    forward = (H.uplo == 'L') == (S === :left)
    entries = collect(B.entries)
    sortrange(k) = S === :left ? W.rowranges[W.entryrows[k]] : W.colranges[csr.colnodes[k]]
    sort!(entries; by=k -> forward ? first(sortrange(k)) : -last(sortrange(k)))
    positions = Dict{Int,Int}()
    for (i, k) in enumerate(entries)
        positions[k] = i
    end
    subfactors = typeof(H)[]
    submap, updatemap = Int[], Int[]
    nodemap = Dict{Int,Int}()
    # Concrete product plan type, also when there are no updates.
    HB = HMatrixBlock{T,typeof(H.workspace),Vector{Int}}
    BB = HMatrixBlock{T,typeof(B.workspace),Vector{Int}}
    products = (S === :left ? _ProductPlan{T,BB,HB,BB} : _ProductPlan{T,BB,BB,HB})[]
    for (step, k) in enumerate(entries)
        rr, cc = _productranges(W, k)
        node = S === :left ? W.entryrows[k] : csr.colnodes[k]
        position = get(nodemap, node, 0)
        if iszero(position)
            push!(subfactors, node == H.node ? H : HMatrixTriangular(H; node))
            position = nodemap[node] = length(subfactors)
        end
        push!(submap, position)
        range = S === :left ? rr : cc
        previous =
            forward ? (first(axis):(first(range) - 1)) : ((last(range) + 1):last(axis))
        update = 0
        if !isempty(previous)
            source =
                S === :left ? _solveblock(W, previous, cc) : _solveblock(W, rr, previous)
            all(j -> 0 < get(positions, j, 0) < step, source.entries) || throw(
                ArgumentError(
                    "unsupported RHS partition crossing: predecessor is not a solved owner",
                ),
            )
            rows, cols = S === :left ? (range, previous) : (previous, range)
            coefficient = _solveblock(H.workspace, rows, cols)
            destination = HMatrixBlock(W, rr, cc, [k])
            A, F = S === :left ? (coefficient, source) : (source, coefficient)
            push!(products, _ProductPlan(destination, A, F; batchrank))
            update = length(products)
        end
        push!(updatemap, update)
    end
    return _SolvePlan{T,typeof(H),typeof(B),typeof(side),eltype(products)}(
        H, B, subfactors, entries, submap, updatemap, products, side
    )
end

function _solveownerfactor!(B, H, scratch, side::Val{S}) where {S}
    rhs = _leafrhs(B, S === :left ? 'L' : 'R')
    isempty(rhs) && return nothing
    width = size(rhs, S === :left ? 2 : 1)
    _growscratch!(scratch.small, checked_mul(_maxrank(H.workspace, H.entries), width))
    _executetriangular!(rhs, H, scratch.small, Val(:N), side)
    return nothing
end

"""
    _executesolveoperation!(P::_SolvePlan, step, scratch, status, options)

Execute step `step` of `P`: subtract the products with the already solved
blocks, recompressing a low-rank block to `options`, and solve with the
diagonal subfactor. The truncation information is added to `status`.
"""
function _executesolveoperation!(P, step, scratch, status, options)
    update = P.updatemap[step]
    if !iszero(update)
        product = P.products[update]
        _executeproduct!(product, -one(eltype(P.rhs.workspace)), scratch, status, options)
    end
    W, csr, k = P.rhs.workspace, P.rhs.workspace.csr, P.entries[step]
    H = P.subfactors[P.submap[step]]
    if csr.isfar[k]
        _solveownerfactor!(W.far[csr.storageidx[k]], H, scratch, P.side)
    else
        _solveownerfactor!(W.near[csr.storageidx[k]], H, scratch, P.side)
    end
    return nothing
end

function _pivotsolveowner!(B, H, inverse)
    rhs = _leafrhs(B, 'L')
    for i in H.leafspan
        node = H.workspace.entryrows[H.leafrecords[i]]
        range = _relative(H.workspace.rowranges[node], H.range)
        _leafpivots!(view(rhs, range, :), H.leaves[i].ipiv, inverse)
    end
    return nothing
end

# Apply the leaf row permutations of the factor to the right-hand side of a
# left-sided plan: P*B; `inverse=true` applies P'.
function _pivot!(P::_SolvePlan; inverse::Bool=false)
    P.side === Val(:left) || throw(ArgumentError("pivots apply to left-sided solves"))
    W, csr = P.rhs.workspace, P.rhs.workspace.csr
    for (step, k) in enumerate(P.entries)
        H = P.subfactors[P.submap[step]]
        if csr.isfar[k]
            _pivotsolveowner!(W.far[csr.storageidx[k]], H, inverse)
        else
            _pivotsolveowner!(W.near[csr.storageidx[k]], H, inverse)
        end
    end
    return P.rhs
end
