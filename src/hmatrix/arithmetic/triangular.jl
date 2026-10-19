# Off-diagonal block of a triangular factor between sibling clusters.
struct _TriangularPanel{T,W<:_TreeView{T}}
    row::Int
    col::Int
    block::HMatrixBlock{T,W,Vector{Int}}
end

"""
    HMatrixTriangular(W::_TreeView, leaves; uplo=:lower)
    HMatrixTriangular(parent::HMatrixTriangular; node)

Lower (unit diagonal) or upper triangular factor of `P*A = L*U` on a view whose
dense diagonal terminals hold the packed LAPACK factors of the leaf LUs
(`leaves`, see [`DenseLeafLU`](@ref)). Both factors share the same view; `L`
reads the strictly lower and `U` the upper part of each leaf. The diagonal
terminals must cover the diagonal without gaps, and no other terminal may cross
it. The row pivots are applied separately with `_pivot!`.

The off-diagonal blocks between sibling clusters are stored as panels with
column (`leftptr`) and row (`rightptr`) adjacency, so a solve can subtract the
contributions of a solved cluster from the clusters still to be solved. The
second form selects the factor of the diagonal cluster `node`, sharing the
panels and maps of `parent`.
"""
struct HMatrixTriangular{T,W<:_TreeView{T}} <: LinearAlgebra.Factorization{T}
    workspace::W
    leaves::Vector{DenseLeafLU{T,Matrix{T}}}
    leafrecords::Vector{Int}
    leafmap::Vector{Int}
    panels::Vector{_TriangularPanel{T,W}}
    leftptr::Vector{Int}
    leftmap::Vector{Int}
    rightptr::Vector{Int}
    rightmap::Vector{Int}
    entries::Vector{Int}
    uplo::Char
    node::Int
    range::UnitRange{Int}
    leafspan::UnitRange{Int}
end

Base.size(H::HMatrixTriangular) = (length(H.range), length(H.range))
Base.size(H::HMatrixTriangular, dim::Integer) = dim <= 2 ? length(H.range) : 1
LinearAlgebra.issuccess(H::HMatrixTriangular) = all(i -> issuccess(H.leaves[i]), H.leafspan)

function _triangularpanels!(panels, W, leafmap, node, uplo)
    !iszero(leafmap[node]) && return nothing
    childnodes = collect(Int, children(testtree(W.tree), node))
    isempty(childnodes) &&
        throw(ArgumentError("missing dense diagonal leaf at cluster $node"))
    for row in childnodes, col in childnodes
        rr, cc = W.rowranges[row], W.colranges[col]
        selected = uplo == 'L' ? last(cc) < first(rr) : last(rr) < first(cc)
        if selected
            block = blockview(W, rr, cc)
            isempty(block.entries) || push!(panels, _TriangularPanel(row, col, block))
        end
    end
    for child in childnodes
        _triangularpanels!(panels, W, leafmap, child, uplo)
    end
    return nothing
end

function HMatrixTriangular(
    W::_TreeView{T}, leaves; uplo::Symbol=:lower, check::Bool=true
) where {T<:LinearAlgebra.BlasFloat}
    uplo in (:lower, :upper) || throw(ArgumentError("uplo must be :lower or :upper"))
    u = uplo === :lower ? 'L' : 'U'
    size(W, 1) == size(W, 2) ||
        throw(DimensionMismatch("triangular factors must be square"))
    testtree(W.tree) === trialtree(W.tree) || throw(
        ArgumentError(
            "triangular factors require the same cluster-tree object on both axes"
        ),
    )
    W.rowperm == W.colperm ||
        throw(ArgumentError("triangular axes have different orderings"))
    csr = W.csr
    leafmap = zeros(Int, length(W.rowranges))
    diagonal = Int[]
    for k in eachindex(csr.storageidx)
        row, col = W.entryrows[k], csr.colnodes[k]
        if row == col
            !csr.isfar[k] || throw(ArgumentError("diagonal cluster blocks must be dense"))
            push!(diagonal, k)
        elseif !isempty(_overlap(W.rowranges[row], W.colranges[col]))
            throw(ArgumentError("terminal block crosses the diagonal cluster frontier"))
        end
    end
    sort!(diagonal; by=k -> first(W.rowranges[W.entryrows[k]]))
    stop = 0
    for k in diagonal
        range = W.rowranges[W.entryrows[k]]
        first(range) == stop + 1 ||
            throw(ArgumentError("diagonal frontier overlaps or has gaps"))
        stop = last(range)
    end
    stop == size(W, 1) || throw(ArgumentError("incomplete dense diagonal frontier"))
    supplied = IdDict{Matrix{T},DenseLeafLU{T,Matrix{T}}}()
    for F in leaves
        F isa DenseLeafLU{T,Matrix{T}} ||
            throw(ArgumentError("leaf factors must reference workspace diagonal matrices"))
        haskey(supplied, F.factors) && throw(ArgumentError("duplicate leaf factorization"))
        supplied[F.factors] = F
    end
    length(supplied) == length(diagonal) ||
        throw(ArgumentError("incorrect number of leaf factorizations"))
    factors = DenseLeafLU{T,Matrix{T}}[]
    for k in diagonal
        node = W.entryrows[k]
        A = W.near[csr.storageidx[k]]
        haskey(supplied, A) ||
            throw(ArgumentError("missing factorization for diagonal cluster $node"))
        F = supplied[A]
        F.node in (0, node) ||
            throw(ArgumentError("leaf factorization has the wrong cluster ID"))
        check && _checkleafstate(F)
        push!(factors, F)
        leafmap[node] = length(factors)
    end
    panels = _TriangularPanel{T,typeof(W)}[]
    _triangularpanels!(panels, W, leafmap, root(testtree(W.tree)), u)
    rows, cols = [p.row for p in panels], [p.col for p in panels]
    leftmap, rightmap = sortperm(cols), sortperm(rows)
    entries = copy(diagonal)
    for p in panels
        append!(entries, p.block.entries)
    end
    sort!(entries)
    unique!(entries)
    return HMatrixTriangular(
        W,
        factors,
        diagonal,
        leafmap,
        panels,
        _compressedptr(cols, length(leafmap)),
        leftmap,
        _compressedptr(rows, length(leafmap)),
        rightmap,
        entries,
        u,
        root(testtree(W.tree)),
        1:size(W, 1),
        1:length(factors),
    )
end

function _triangularleafposition(H, position)
    lo, hi = first(H.leafspan), last(H.leafspan) + 1
    while lo < hi
        mid = (lo + hi) >>> 1
        start = first(H.workspace.rowranges[H.workspace.entryrows[H.leafrecords[mid]]])
        if start < position
            lo = mid + 1
        else
            hi = mid
        end
    end
    return lo
end

function HMatrixTriangular(H::HMatrixTriangular; node::Integer)
    W, csr = H.workspace, H.workspace.csr
    1 <= node <= length(W.rowranges) || throw(BoundsError(W.rowranges, node))
    range = W.rowranges[node]
    _overlap(range, H.range) == range ||
        throw(ArgumentError("cluster is outside the parent factor view"))
    firstleaf = _triangularleafposition(H, first(range))
    lastleaf = _triangularleafposition(H, last(range) + 1) - 1
    firstleaf <= lastleaf &&
    first(W.rowranges[W.entryrows[H.leafrecords[firstleaf]]]) == first(range) &&
    last(W.rowranges[W.entryrows[H.leafrecords[lastleaf]]]) == last(range) ||
        throw(ArgumentError("cluster selection cuts a dense diagonal terminal"))
    # Unary clusters can share a range: a selection of exactly one leaf executes
    # at that leaf, not at a same-range ancestor.
    selectednode = firstleaf == lastleaf ? W.entryrows[H.leafrecords[firstleaf]] : Int(node)
    block = blockview(W, range, range)
    entries = filter(block.entries) do k
        row, col = W.entryrows[k], csr.colnodes[k]
        return row == col || (
            if H.uplo == 'L'
                last(W.colranges[col]) < first(W.rowranges[row])
            else
                last(W.rowranges[row]) < first(W.colranges[col])
            end
        )
    end
    return HMatrixTriangular(
        W,
        H.leaves,
        H.leafrecords,
        H.leafmap,
        H.panels,
        H.leftptr,
        H.leftmap,
        H.rightptr,
        H.rightmap,
        entries,
        H.uplo,
        selectednode,
        range,
        firstleaf:lastleaf,
    )
end

function _checktriangularrhs(B::StridedMatrix, H, side)
    Base.require_one_based_indexing(B)
    size(B, side === :left ? 1 : 2) == size(H, 1) ||
        throw(DimensionMismatch("RHS does not match triangular factor"))
    isempty(B) || _lapackmatrix(B)
    return nothing
end

@inline _triangularslice(B, range, ::Val{:left}) = view(B, range, :)
@inline _triangularslice(B, range, ::Val{:right}) = view(B, :, range)

# Solve for one child cluster, then subtract its contributions to the clusters
# still to be solved.
function _triangularchild!(
    B, H, child, buffer, op::Val{O}, side::Val{S}, forward
) where {O,S}
    _triangularnode!(B, H, child, buffer, op, side, forward)
    lead = (O === :N) == (S === :left)
    ptr, map = lead ? (H.leftptr, H.leftmap) : (H.rightptr, H.rightmap)
    for j in ptr[child]:(ptr[child + 1] - 1)
        panel = H.panels[map[j]]
        rows = _relative(H.workspace.rowranges[panel.row], H.range)
        cols = _relative(H.workspace.colranges[panel.col], H.range)
        target, source = lead ? (rows, cols) : (cols, rows)
        Y, X = _triangularslice(B, target, side), _triangularslice(B, source, side)
        _thinexecute!(Y, X, panel.block, -one(eltype(B)), buffer, op, side)
    end
    return nothing
end

# The cluster child iterator is forward-only; reverse order recurses on its states.
function _triangularreverse!(B, H, children, info, buffer, op, side, forward)
    isnothing(info) && return nothing
    child, state = info
    _triangularreverse!(B, H, children, iterate(children, state), buffer, op, side, forward)
    _triangularchild!(B, H, child, buffer, op, side, forward)
    return nothing
end

function _triangularnode!(
    B, H, node, buffer, op::Val{O}, side::Val{S}, forward::Val{F}
) where {O,S,F}
    leaf = H.leafmap[node]
    if !iszero(leaf)
        rhs = _triangularslice(B, _relative(H.workspace.rowranges[node], H.range), side)
        isempty(rhs) || BLAS.trsm!(
            S === :left ? 'L' : 'R',
            H.uplo,
            O === :N ? 'N' : 'T',
            H.uplo == 'L' ? 'U' : 'N',
            one(eltype(B)),
            H.leaves[leaf].factors,
            rhs,
        )
        return nothing
    end
    childnodes = children(testtree(H.workspace.tree), node)
    if F
        for child in childnodes
            _triangularchild!(B, H, child, buffer, op, side, forward)
        end
    else
        _triangularreverse!(
            B, H, childnodes, iterate(childnodes), buffer, op, side, forward
        )
    end
    return nothing
end

function _executetriangular!(B, H, buffer, op::Val{O}, side::Val{S}) where {O,S}
    forward = (H.uplo == 'L') == (S === :left)
    O === :T && (forward = !forward)
    node = H.node
    # Branch before calling: runtime Val(forward) boxes this immutable plan.
    if forward
        _triangularnode!(B, H, node, buffer, op, side, Val(true))
    else
        _triangularnode!(B, H, node, buffer, op, side, Val(false))
    end
    return B
end

"""
    _trisolve!(B::StridedMatrix, H::HMatrixTriangular, scratch, Val(op), Val(side))

Solve `op(H)*X = B` (`side=:left`) or `X*op(H) = B` (`side=:right`) in place for
a dense right-hand side in tree order, with `op` `:N` or `:T`. The solve
recurses over the cluster tree: leaves use BLAS `trsm`, and after each child
cluster its panels update the remaining part of `B` with thin products. The
pivots are not applied.
"""
function _trisolve!(
    B::StridedMatrix, H::HMatrixTriangular{T}, scratch, ::Val{O}, side::Val{S}
) where {T,O,S}
    _checktriangularrhs(B, H, S)
    foreach(i -> _checkleafstate(H.leaves[i]), H.leafspan)
    isempty(B) && return B
    width = size(B, S === :left ? 2 : 1)
    _growscratch!(scratch.small, checked_mul(_maxrank(H.workspace, H.entries), width))
    _executetriangular!(B, H, scratch.small, Val(O), side)
    return B
end

# Apply the disjoint diagonal-leaf row permutations in tree order: P*B;
# `inverse=true` applies P'.
function _pivot!(
    B::StridedMatrix{T}, H::HMatrixTriangular{T}; inverse::Bool=false
) where {T<:LinearAlgebra.BlasFloat}
    _checktriangularrhs(B, H, :left)
    W = H.workspace
    for i in H.leafspan
        node = W.entryrows[H.leafrecords[i]]
        rows = _relative(W.rowranges[node], H.range)
        _leafpivots!(view(B, rows, :), H.leaves[i].ipiv, inverse)
    end
    return B
end
