# Cached numerical-panel references, not a second cluster/block hierarchy.
struct _TriangularPanel{T,W<:_TreeView{T}}
    row::Int
    col::Int
    block::HMatrixBlock{T,W,Vector{Int}}
end

# Read-only triangular factor view and reusable execution index for the
# convention P*A = L*U. The view holds combined LU data: dense diagonal terminals
# contain packed leaf LU, strictly lower/upper terminals the factors. `leaves`
# are the DenseLeafLU objects of exactly these diagonal matrices. `uplo=:lower`
# has unit diagonal, `:upper` nonunit diagonal. Pivots are never implicit: use
# `_pivot!` explicitly. Arithmetic is in tree order.
#
# Sibling panel views and their integer execution maps are built once.
# `HMatrixTriangular(parent; node)` selects a diagonal cluster subtree sharing the
# parent's maps and leaf objects. Coefficients are read-only during solves;
# concurrent solves need independent right-hand sides and scratch.
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
Base.size(H::HMatrixTriangular, dim::Integer) =
    if dim in (1, 2)
        length(H.range)
    elseif dim > 2
        1
    else
        throw(ArgumentError("dimension must be positive"))
    end
LinearAlgebra.issuccess(H::HMatrixTriangular) = all(i -> issuccess(H.leaves[i]), H.leafspan)

function _triangularpanels!(panels, W, leafmap, node, uplo)
    !iszero(leafmap[node]) && return nothing
    children = collect(Int, _clusterchildren(testtree(W.tree), node))
    isempty(children) &&
        throw(ArgumentError("missing dense diagonal leaf at cluster $node"))
    for row in children, col in children
        rr, cc = W.rowranges[row], W.colranges[col]
        selected = uplo == 'L' ? last(cc) < first(rr) : last(rr) < first(cc)
        if selected
            block = blockview(W, rr, cc)
            isempty(block.entries) || push!(panels, _TriangularPanel(row, col, block))
        end
    end
    for child in children
        _triangularpanels!(panels, W, leafmap, child, uplo)
    end
    return nothing
end

function HMatrixTriangular(
    W::_TreeView{T}, leaves; uplo::Symbol=:lower, check::Bool=true
) where {T<:LinearAlgebra.BlasFloat}
    u = _leafuplo(uplo)
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
        _checkworkspaceblock(W, k)
        if check
            _checkleafstate(F)
            _checkleafpivots(F)
        end
        push!(factors, F)
        leafmap[node] = length(factors)
    end
    panels = _TriangularPanel{T,typeof(W)}[]
    _triangularpanels!(panels, W, leafmap, _clusterroot(testtree(W.tree)), u)
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
        _clusterroot(testtree(W.tree)),
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
    # Unary ancestors/descendants can have identical ranges. If selection is
    # exactly one physical diagonal leaf, execute at that leaf rather than
    # descending past its cached factorization through a same-range descendant.
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
    # Global execution maps are shared, not rebuilt for every diagonal subtree.
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

# One preflight for the complete operation. Numerical recursion never validates
# whole subworkspaces again, constructs blockviews, or changes scratch capacity.
function _checktriangular(B, H, scratch, side)
    _checktriangularrhs(B, H, side)
    foreach(buffer -> _checkproductalias(buffer, B), _scratcharrays(scratch))
    W, csr = H.workspace, H.workspace.csr
    for i in H.leafspan
        F = H.leaves[i]
        F.factors === W.near[csr.storageidx[H.leafrecords[i]]] || throw(
            ArgumentError("diagonal storage was replaced; rebuild the triangular plan")
        )
        _checkleafstate(F)
    end
    maximumrank = 0
    for k in H.entries
        _checkworkspaceblock(W, k)
        if csr.isfar[k]
            R = W.far[csr.storageidx[k]]
            _checkproductalias(B, R)
            foreach(buffer -> _checkproductalias(buffer, R), _scratcharrays(scratch))
            maximumrank = max(maximumrank, rank(R))
        else
            D = W.near[csr.storageidx[k]]
            _checkproductalias(B, D)
            foreach(buffer -> _checkproductalias(buffer, D), _scratcharrays(scratch))
        end
    end
    return maximumrank
end

@inline _triangularslice(B, range, ::Val{:left}) = view(B, range, :)
@inline _triangularslice(B, range, ::Val{:right}) = view(B, :, range)

function _triangularupdate!(B, H, panel, buffer, op::Val{O}, side::Val{S}) where {O,S}
    rows = _relative(H.workspace.rowranges[panel.row], H.range)
    cols = _relative(H.workspace.colranges[panel.col], H.range)
    target, source = if (O === :N) == (S === :left)
        rows, cols
    else
        cols, rows
    end
    Y, X = _triangularslice(B, target, side), _triangularslice(B, source, side)
    _thinexecute!(Y, X, panel.block, -one(eltype(B)), buffer, op, side)
    return nothing
end

function _triangularchild!(
    B, H, child, buffer, op::Val{O}, side::Val{S}, forward
) where {O,S}
    _triangularnode!(B, H, child, buffer, op, side, forward)
    ptr, map =
        (O === :N) == (S === :left) ? (H.leftptr, H.leftmap) : (H.rightptr, H.rightmap)
    for j in ptr[child]:(ptr[child + 1] - 1)
        _triangularupdate!(B, H, H.panels[map[j]], buffer, op, side)
    end
    return nothing
end

# H2Trees' linked child iterator is forward-only. Reverse visitation uses the
# iterator's own sibling states on the stack; no copied child hierarchy/vector.
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
    children = _clusterchildren(testtree(H.workspace.tree), node)
    if F
        for child in children
            _triangularchild!(B, H, child, buffer, op, side, forward)
        end
    else
        _triangularreverse!(B, H, children, iterate(children), buffer, op, side, forward)
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

# In-place triangular solves op(H)*X = B (left) or X*op(H) = B (right) with a
# dense column-major RHS. One preflight validates shapes, factor state and
# aliases before writes.
function _trisolve!(
    B::StridedMatrix, H::HMatrixTriangular{T}, scratch, ::Val{O}, side::Val{S}
) where {T,O,S}
    r = _checktriangular(B, H, scratch, S)
    isempty(B) && return B
    width = size(B, S === :left ? 2 : 1)
    _growscratch!(scratch.small, checked_mul(r, width))
    _executetriangular!(B, H, scratch.small, Val(O), side)
    return B
end

# Apply the disjoint diagonal-leaf row permutations in tree order: P*B;
# `inverse=true` applies P'.
function _pivot!(
    B::StridedMatrix{T}, H::HMatrixTriangular{T}; inverse::Bool=false
) where {T<:LinearAlgebra.BlasFloat}
    _checktriangularrhs(B, H, :left)
    W, csr = H.workspace, H.workspace.csr
    for i in H.leafspan
        F = H.leaves[i]
        F.factors === W.near[csr.storageidx[H.leafrecords[i]]] ||
            throw(ArgumentError("diagonal storage was replaced"))
        _checkleafstate(F)
        _checkleafpivots(F)
    end
    # Reject overlap with ALL touched coefficients, not only diagonal leaves.
    for k in H.entries
        _checkproductalias(
            B, csr.isfar[k] ? W.far[csr.storageidx[k]] : W.near[csr.storageidx[k]]
        )
    end
    for i in H.leafspan
        node = W.entryrows[H.leafrecords[i]]
        rows = _relative(W.rowranges[node], H.range)
        _leafpivots!(view(B, rows, :), H.leaves[i].ipiv, inverse)
    end
    return B
end
