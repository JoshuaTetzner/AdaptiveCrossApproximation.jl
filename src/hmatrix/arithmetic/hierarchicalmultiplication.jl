"""
    ProductWorkspace(C::HMatrixCore, A::HMatrixCore, B::HMatrixCore;
                     batchrank=128, ntasks=Threads.nthreads())

Plan and buffers for `mul!(C, A, B, α, β, workspace)`. Reusable for every triple
of matrices with the same partitions as `C`, `A` and `B`. The blocks of `C` are
updated on `ntasks` parallel tasks; the result does not depend on `ntasks`, and
`ntasks=1` runs on the calling task. `batchrank` bounds the pending update
columns per far block.
"""
struct ProductWorkspace{T,VC,VA,VB,P,R}
    c::VC
    a::VA
    b::VB
    plan::P
    graph::_TaskGraph
    workers::Vector{_Worker{T,R}}
end

function ProductWorkspace(
    C::HMatrixCore,
    A::HMatrixCore,
    B::HMatrixCore;
    batchrank::Integer=128,
    ntasks::Integer=Threads.nthreads(),
)
    T = _checkeltypes(C, A, B)
    c, a, b = _TreeView(C), _TreeView(A), _TreeView(B)
    plan = _ProductPlan(blockview(c), blockview(a), blockview(b); batchrank)
    workers = [_Worker{T}() for _ in 1:_checkntasks(ntasks)]
    return ProductWorkspace(c, a, b, plan, _productgraph(plan), workers)
end

"""
    mul!(C::HMatrixCore, A::HMatrixCore, B::HMatrixCore, α=true, β=false[, workspace])

Overwrite `C` with `α*A*B + β*C`, recompressed to the largest tolerance of the
three matrices. `A` and `B` are not modified and must not be `C`. The row
cluster tree of `C` is that of `A`, its column tree that of `B`, and the column
tree of `A` the row tree of `B`; the block partitions may differ.
"""
function LinearAlgebra.mul!(
    C::HMatrixCore,
    A::HMatrixCore,
    B::HMatrixCore,
    alpha::Number,
    beta::Number,
    ws::ProductWorkspace,
)
    _checkeltypes(C, A, B)
    (C === A || C === B) &&
        throw(ArgumentError("the destination of mul! must not be one of its factors"))
    _bind!(ws.c, C)
    _bind!(ws.a, A)
    _bind!(ws.b, B)
    rmul!(C, beta)
    _addmul!(ws.plan, alpha, ws.workers, ws.graph, _options(C, A, B))
    return C
end
LinearAlgebra.mul!(
    C::HMatrixCore, A::HMatrixCore, B::HMatrixCore, alpha::Number, beta::Number
) = mul!(C, A, B, alpha, beta, ProductWorkspace(C, A, B))
LinearAlgebra.mul!(C::HMatrixCore, A::HMatrixCore, B::HMatrixCore) =
    mul!(C, A, B, true, false)

"""
    A * B

Product of two H-matrices as a new `HMatrixCore` with the block partition of `A`
and the larger tolerance of the two.
"""
function Base.:*(A::HMatrixCore, B::HMatrixCore)
    T = _checkeltypes(A, B)
    size(A, 2) == size(B, 1) || throw(DimensionMismatch("incompatible matrix sizes"))
    near = [zeros(T, size(D)) for D in A.near]
    far = [
        LowRankMatrix{T}(zeros(T, size(R, 1), 0), zeros(T, 0, size(R, 2))) for R in A.far
    ]
    C = HMatrix{T}(A.tree, near, far, A.csr, A.csc, A.ordering, A.dim, max(A.tol, B.tol))
    return mul!(C, A, B, true, false)
end

"""
    _ProductPanel

A low-rank terminal `entry` of one operand together with the hierarchical panel
of the other operand it multiplies: the terminals `entries`, clipped to
`rows × cols`. A far terminal `U*V` of `A` contributes `U * (V*B[panel])`, one
thin product with the whole panel instead of one product per terminal pair.
"""
struct _ProductPanel
    entry::Int
    rows::UnitRange{Int}
    cols::UnitRange{Int}
    entries::Vector{Int}
end

"""
    _ProductGroup

All contributions to one destination terminal `destination` of a product,
restricted to `rows × cols`: the terminals of `A` and `B` it reads (`aentries`,
`bentries`), the low-rank panels it multiplies itself (`afar`, `bfar`) and its
nearest [`_ProductNode`](@ref) (`node`, zero if none). The execution strategy is
fixed at plan time (`implicit`) or chosen from the current ranks before each
execution (`capacity` of pending update columns, `dense` core); `denserank` is
the summed inner dimension of its dense/dense terminal pairs.
"""
mutable struct _ProductGroup
    destination::Int
    rows::UnitRange{Int}
    cols::UnitRange{Int}
    aentries::Vector{Int}
    bentries::Vector{Int}
    bmaxend::Vector{Int}
    afar::Vector{_ProductPanel}
    bfar::Vector{_ProductPanel}
    denserank::Float64
    capacity::Int
    dense::Bool
    node::Int
    implicit::Bool
end

"""
    _ProductNode

Virtual block (row cluster × column cluster) above at least two complete
destinations of a product. It collects the low-rank panels whose terminal
strictly contains these destinations, forms their product once, adds the slice
of its parent node and is recompressed once. `estimate` is its current rank
bound, used to size the buffers of the destinations below it. See
[`_ProductPlan`](@ref).
"""
mutable struct _ProductNode
    rows::UnitRange{Int}
    cols::UnitRange{Int}
    parent::Int
    afar::Vector{_ProductPanel}
    bfar::Vector{_ProductPanel}
    estimate::Int
end

"""
    _ProductPlan(C::HMatrixBlock, A::HMatrixBlock, B::HMatrixBlock; batchrank=128)

Reusable plan for `C += α*A*B` on hierarchical blocks whose corresponding axes
share cluster trees; the terminal partitions may differ. Ranks may change
between executions, the partitions may not. It is the kernel of `mul!`, of the
Schur-complement updates of the LU factorization and of the updates in the
triangular solves.

The plan has one [`_ProductGroup`](@ref) per destination terminal of `C`, which
collects every terminal pair of `A` and `B` meeting it:

  - dense × dense pairs contribute exact factor slices, or are summed into a
    dense `m × n` core once their inner dimensions reach `mn/(m+n)`;
  - a low-rank terminal multiplies the whole hierarchical panel of the other
    operand at once ([`_ProductPanel`](@ref));
  - a low-rank terminal whose rows (or columns) strictly contain several
    complete destinations is attached to a virtual [`_ProductNode`](@ref) above
    them: its product is formed and recompressed once, and the destinations
    receive a slice of their nearest node;
  - a complete far destination is recompressed directly from the implicit
    operator `C + α*A*B` ([`_ProductOperator`](@ref)) when that is estimated to
    be cheaper.

Low-rank updates are collected as pending columns, at most `batchrank` per far
destination, and recompressed together with the destination. Buffers belong
to the executing task's scratch, so nodes and destinations can run on
different tasks; nodes run before the nodes and destinations below them (see
`_productgraph`).
"""
mutable struct _ProductPlan{T,C,A,B}
    destination::C
    a::A
    b::B
    groups::Vector{_ProductGroup}
    nodes::Vector{_ProductNode}
    results::Vector{LowRankMatrix{T}}
    batchrank::Int
end

function _productranges(W, k)
    return W.rowranges[W.entryrows[k]], W.colranges[W.csr.colnodes[k]]
end

function _intervalmax!(ends, entries, W, node, lo, hi)
    if lo == hi
        ends[node] = last(first(_productranges(W, entries[lo])))
    else
        mid = (lo + hi) >>> 1
        ends[node] = max(
            _intervalmax!(ends, entries, W, 2node, lo, mid),
            _intervalmax!(ends, entries, W, 2node + 1, mid + 1, hi),
        )
    end
    return ends[node]
end

# Number of parent steps from `node` to the strict ancestor `target`, or 0.
function _ancestorsteps(parents, node, target)
    steps = 0
    while node != 0 && node != target
        node = parents[node]
        steps += 1
    end
    return node == target ? steps : 0
end

function _ancestor(parents, node, steps)
    for _ in 1:steps
        node == 0 && return 0
        node = parents[node]
    end
    return node
end

# Aggregation key of a low-rank source term for destination (τ, σ), or nothing.
# Rows: the A terminal's row cluster τa must be a strict ancestor of τ; the key
# pairs it with σ's ancestor the same number of levels up. Columns mirror this.
function _productkey(rowparents, colparents, τ, σ, τa, σb)
    if τa != 0
        steps = _ancestorsteps(rowparents, τ, τa)
        steps == 0 && return nothing
        σk = _ancestor(colparents, σ, steps)
        return σk == 0 ? nothing : (τa, σk)
    end
    steps = _ancestorsteps(colparents, σ, σb)
    steps == 0 && return nothing
    τk = _ancestor(rowparents, τ, steps)
    return τk == 0 ? nothing : (τk, σb)
end

# Nearest strict ancestor pair of (τ, σ) present in `keys`, or 0.
function _nearestproductnode(keys, rowparents, colparents, τ, σ)
    while true
        τ, σ = rowparents[τ], colparents[σ]
        (τ == 0 || σ == 0) && return 0
        node = get(keys, (τ, σ), 0)
        node == 0 || return node
    end
end

# Stored entries of the terminals in `entries`, clipped to logical rows × cols.
function _panelstorage(H, entries, rows, cols)
    W, total = H.workspace, 0.0
    for k in entries
        rr, cc = _productranges(W, k)
        lr, lc = length(_overlap(rr, rows)), length(_overlap(cc, cols))
        total += if W.csr.isfar[k]
            rank(W.far[W.csr.storageidx[k]]) * (lr + lc)
        else
            Float64(lr) * lc
        end
    end
    return total
end

"""
    _implicitcheaper(C, A, B, k, rows, cols, ai, bi, denserank, parents)

Plan-time flop model for the complete far destination `k`, with the ranks at
planning time. Implicit: every sketch column passes once through
`B(inner, cols)` and `A(rows, inner)`. Factors: each local low-rank term
(ancestor terms are shared through nodes) costs a thin product plus its pending
columns in the sketch, and dense pairs enter as factor slices or a dense core.
Returns whether the implicit operator is cheaper.
"""
function _implicitcheaper(C, A, B, k, rows, cols, ai, bi, denserank, parents)
    W = C.workspace
    m, n = length(rows), length(cols)
    width = 2 * rank(W.far[W.csr.storageidx[k]]) + 18
    inner = A.cols
    implicit =
        width * (_panelstorage(A, ai, rows, inner) + _panelstorage(B, bi, inner, cols))
    rowparents, colparents = parents
    τ, σ = W.entryrows[k], W.csr.colnodes[k]
    factors = min(denserank, m * n / (m + n)) * (m + n) * width
    for i in ai
        A.workspace.csr.isfar[i] || continue
        _ancestorsteps(rowparents, τ, A.workspace.entryrows[i]) == 0 || continue
        r = rank(A.workspace.far[A.workspace.csr.storageidx[i]])
        _, ac = _productranges(A.workspace, i)
        interval = _overlap(ac, inner)
        entries = _productsubentries(B, interval, cols)
        factors += r * (_panelstorage(B, entries, interval, cols) + (m + n) * width)
        factors > implicit && return true
    end
    for j in bi
        B.workspace.csr.isfar[j] || continue
        _ancestorsteps(colparents, σ, B.workspace.csr.colnodes[j]) == 0 || continue
        r = rank(B.workspace.far[B.workspace.csr.storageidx[j]])
        br, _ = _productranges(B.workspace, j)
        interval = _overlap(br, inner)
        entries = filter(
            i -> !A.workspace.csr.isfar[i], _productsubentries(A, rows, interval)
        )
        factors += r * (_panelstorage(A, entries, rows, interval) + (m + n) * width)
        factors > implicit && return true
    end
    return factors > implicit
end

"""
    _productgroups(C, A, B)

Analyse the terminals of a product: the [`_ProductGroup`](@ref) of every
destination of `C`, the candidate virtual nodes kept where they serve at least
two complete destinations (ordered parents first), and the assignment of every
low-rank terminal to its node or to the panels of its destination. Returns
`(groups, nodes)`.
"""
function _productgroups(C, A, B)
    groups = _ProductGroup[]
    W = C.workspace
    rowtree, coltree = testtree(W.tree), trialtree(W.tree)
    rowparents = [parent(rowtree, node) for node in 1:numberofnodes(rowtree)]
    colparents = if coltree === rowtree
        rowparents
    else
        [parent(coltree, node) for node in 1:numberofnodes(coltree)]
    end
    inner = A.cols
    # Only complete destinations receive node slices or use the implicit operator.
    whole, implicit = Bool[], Bool[]
    for k in C.entries
        rr, cc = _productranges(W, k)
        rows, cols = _overlap(rr, C.rows), _overlap(cc, C.cols)
        ai = _productsubentries(A, rows, inner)
        bi = _productsubentries(B, B.rows, cols)
        sort!(bi; by=j -> first(first(_productranges(B.workspace, j))))
        ends = zeros(Int, 4length(bi))
        isempty(bi) || _intervalmax!(ends, bi, B.workspace, 1, 1, length(bi))
        denserank = 0.0
        for i in ai
            A.workspace.csr.isfar[i] && continue
            _, ac = _productranges(A.workspace, i)
            denserank += _densejoinrank(B, bi, ends, _overlap(ac, inner), 1, 1, length(bi))
        end
        complete = rows == rr && cols == cc
        operator =
            complete &&
            W.csr.isfar[k] &&
            _implicitcheaper(
                C, A, B, k, rows, cols, ai, bi, denserank, (rowparents, colparents)
            )
        push!(implicit, operator)
        push!(whole, complete && !operator)
        push!(
            groups,
            _ProductGroup(
                k,
                rows,
                cols,
                ai,
                bi,
                ends,
                _ProductPanel[],
                _ProductPanel[],
                denserank,
                0,
                false,
                0,
                implicit[end],
            ),
        )
    end
    # Candidate nodes, kept only when at least two destinations lie below them:
    # a node serving one destination would only add a recompression.
    candidates = Dict{Tuple{Int,Int},Int}()
    for (g, complete) in zip(groups, whole)
        complete || continue
        τ, σ = W.entryrows[g.destination], W.csr.colnodes[g.destination]
        for i in g.aentries
            A.workspace.csr.isfar[i] || continue
            key = _productkey(rowparents, colparents, τ, σ, A.workspace.entryrows[i], 0)
            isnothing(key) || (candidates[key] = 0)
        end
        for j in g.bentries
            B.workspace.csr.isfar[j] || continue
            key = _productkey(rowparents, colparents, τ, σ, 0, B.workspace.csr.colnodes[j])
            isnothing(key) || (candidates[key] = 0)
        end
    end
    for (g, complete) in zip(groups, whole)
        (complete && !isempty(candidates)) || continue
        τ, σ = W.entryrows[g.destination], W.csr.colnodes[g.destination]
        while true
            τ, σ = rowparents[τ], colparents[σ]
            (τ == 0 || σ == 0) && break
            haskey(candidates, (τ, σ)) && (candidates[(τ, σ)] += 1)
        end
    end
    keys = Dict{Tuple{Int,Int},Int}()
    nodes = _ProductNode[]
    depths = Int[]
    for (key, cover) in candidates
        cover >= 2 || continue
        rows = _overlap(W.rowranges[key[1]], C.rows)
        cols = _overlap(W.colranges[key[2]], C.cols)
        push!(nodes, _ProductNode(rows, cols, 0, _ProductPanel[], _ProductPanel[], 0))
        push!(depths, _ancestorsteps(rowparents, key[1], root(rowtree)))
        keys[key] = length(nodes)
    end
    # Parents before children; renumber the keys accordingly.
    order = sortperm(depths)
    nodes = nodes[order]
    position = invperm(order)
    for key in collect(Base.keys(keys))
        keys[key] = position[keys[key]]
    end
    nodekeys = Vector{Tuple{Int,Int}}(undef, length(nodes))
    for (key, n) in keys
        nodekeys[n] = key
    end
    for (n, key) in enumerate(nodekeys)
        nodes[n].parent = _nearestproductnode(keys, rowparents, colparents, key...)
    end
    aseen, bseen = Set{Tuple{Int,Int}}(), Set{Tuple{Int,Int}}()
    for (g, complete) in zip(groups, whole)
        rows, cols = g.rows, g.cols
        τ, σ = W.entryrows[g.destination], W.csr.colnodes[g.destination]
        if complete && !isempty(nodes)
            g.node = _nearestproductnode(keys, rowparents, colparents, τ, σ)
        end
        # The remaining low-rank terminals multiply whole panels.
        for i in g.aentries
            A.workspace.csr.isfar[i] || continue
            _, ac = _productranges(A.workspace, i)
            interval = _overlap(ac, inner)
            key = if complete && !isempty(nodes)
                _productkey(rowparents, colparents, τ, σ, A.workspace.entryrows[i], 0)
            else
                nothing
            end
            n = isnothing(key) ? 0 : get(keys, key, 0)
            if n == 0
                entries = _productsubentries(B, interval, cols)
                push!(g.afar, _ProductPanel(i, interval, cols, entries))
            elseif !((n, i) in aseen)
                push!(aseen, (n, i))
                entries = _productsubentries(B, interval, nodes[n].cols)
                push!(nodes[n].afar, _ProductPanel(i, interval, nodes[n].cols, entries))
            end
        end
        for j in g.bentries
            B.workspace.csr.isfar[j] || continue
            br, _ = _productranges(B.workspace, j)
            interval = _overlap(br, inner)
            key = if complete && !isempty(nodes)
                _productkey(rowparents, colparents, τ, σ, 0, B.workspace.csr.colnodes[j])
            else
                nothing
            end
            n = isnothing(key) ? 0 : get(keys, key, 0)
            target = n == 0 ? rows : nodes[n].rows
            n != 0 && (n, j) in bseen && continue
            n == 0 || push!(bseen, (n, j))
            entries = _productsubentries(A, target, interval)
            filter!(i -> !A.workspace.csr.isfar[i], entries)
            isempty(entries) && continue
            panel = _ProductPanel(j, target, interval, entries)
            push!(n == 0 ? g.bfar : nodes[n].bfar, panel)
        end
    end
    return groups, nodes
end

function _productrankestimate(P, g)
    estimated = g.denserank
    g.node == 0 || (estimated += P.nodes[g.node].estimate)
    for panel in g.afar
        r = rank(P.a.workspace.far[P.a.workspace.csr.storageidx[panel.entry]])
        if _singlefar(P.b, panel)
            r = min(
                r,
                rank(P.b.workspace.far[P.b.workspace.csr.storageidx[only(panel.entries)]]),
            )
        end
        estimated += r
    end
    for panel in g.bfar
        estimated += rank(P.b.workspace.far[P.b.workspace.csr.storageidx[panel.entry]])
    end
    return estimated
end

_singlefar(H, panel) =
    length(panel.entries) == 1 && H.workspace.csr.isfar[only(panel.entries)]

function _densejoinrank(B, entries, ends, interval, node, lo, hi)
    lo > hi && return 0.0
    ends[node] < first(interval) && return 0.0
    br, _ = _productranges(B.workspace, entries[lo])
    first(br) > last(interval) && return 0.0
    if lo == hi
        B.workspace.csr.isfar[entries[lo]] && return 0.0
        return Float64(length(_overlap(br, interval)))
    end
    mid = (lo + hi) >>> 1
    return _densejoinrank(B, entries, ends, interval, 2node, lo, mid) +
           _densejoinrank(B, entries, ends, interval, 2node + 1, mid + 1, hi)
end

_subblock(H, rows, cols, entries) =
    HMatrixBlock{eltype(H),typeof(H.workspace),typeof(entries)}(
        H.workspace, rows, cols, entries
    )
_panelblock(H, panel) = _subblock(H, panel.rows, panel.cols, panel.entries)

# Factor views of the far terminal k of H, restricted to rows × cols.
function _farfactors(H, k, rows, cols)
    W = H.workspace
    R = W.far[W.csr.storageidx[k]]
    return view(R.U, _relative(rows, _rowrange(W, k)), :),
    view(R.V, :, _relative(cols, _colrange(W, k)))
end

function _writepanelchunk!(U, V, X, Y, H, columns, alpha, scratch, ::Val{:left})
    copyto!(U, view(X, :, columns))
    _thinright!(V, view(Y, columns, :), H, alpha, zero(alpha), scratch)
    return nothing
end
function _writepanelchunk!(U, V, X, Y, H, columns, alpha, scratch, ::Val{:right})
    _thinleft!(U, H, view(X, :, columns), alpha, zero(alpha), scratch)
    copyto!(V, view(Y, columns, :))
    return nothing
end

function _productpanelchunks!(
    P, g, X, Y, H, rows, cols, alpha, scratch, pending, status, options, side
)
    W, csr = P.destination.workspace, P.destination.workspace.csr
    k = g.destination
    rr, cc = _productranges(W, k)
    rrows, rcols = _relative(rows, rr), _relative(cols, cc)
    offset, total = 0, size(X, 2)
    while offset < total
        if pending == g.capacity
            _flushproduct!(P, g, pending, scratch, status, options)
            pending = 0
        end
        r = min(total - offset, g.capacity - pending)
        U, V = _productpending(P, g, pending + r, scratch)
        up = view(U, :, (pending + 1):(pending + r))
        vp = view(V, (pending + 1):(pending + r), :)
        fill!(up, zero(eltype(up)))
        fill!(vp, zero(eltype(vp)))
        _writepanelchunk!(
            view(up, rrows, :),
            view(vp, :, rcols),
            X,
            Y,
            H,
            (offset + 1):(offset + r),
            alpha,
            scratch,
            side,
        )
        pending += r
        offset += r
    end
    return pending
end

function _productsubentries(H, rows, cols)
    # Small panels (typical for solve updates) are scanned directly.
    if length(H.entries) <= 64
        entries = Int[]
        for k in H.entries
            rr, cc = _productranges(H.workspace, k)
            isempty(_overlap(rr, rows)) || isempty(_overlap(cc, cols)) || push!(entries, k)
        end
        return entries
    end
    entries = blockview(H.workspace, rows, cols).entries
    return entries isa Vector{Int} ? entries : collect(Int, entries)
end

function _ProductPlan(
    c::HMatrixBlock{T}, a::HMatrixBlock{T}, b::HMatrixBlock{T}; batchrank::Integer=128
) where {T<:LinearAlgebra.BlasFloat}
    batchrank > 0 || throw(ArgumentError("batchrank must be positive"))
    cw, aw, bw = c.workspace, a.workspace, b.workspace
    testtree(cw.tree) === testtree(aw.tree) &&
    trialtree(cw.tree) === trialtree(bw.tree) &&
    trialtree(aw.tree) === testtree(bw.tree) ||
        throw(ArgumentError("product axes require matching cluster-tree objects"))
    cw.rowperm == aw.rowperm && cw.colperm == bw.colperm && aw.colperm == bw.rowperm ||
        throw(ArgumentError("product axes have different external orderings"))
    c.rows == a.rows && c.cols == b.cols && a.cols == b.rows ||
        throw(DimensionMismatch("product axes require matching tree-order ranges"))
    groups, nodes = _productgroups(c, a, b)
    return _ProductPlan{T,typeof(c),typeof(a),typeof(b)}(
        c, a, b, groups, nodes, LowRankMatrix{T}[], Int(batchrank)
    )
end

# Grow the product buffers of `scratch` to exactly the requested lengths; no
# buffer views may be live.
function _reserveproduct!(scratch::BlockScratch{T}, left, right, aggregate) where {T}
    left > length(scratch.productleft) && (scratch.productleft = Vector{T}(undef, left))
    right > length(scratch.productright) && (scratch.productright = Vector{T}(undef, right))
    aggregate > length(scratch.aggregate) &&
        (scratch.aggregate = Vector{T}(undef, aggregate))
    return nothing
end

# Nodes and destinations size their buffers from the current ranks of their
# operands just before they run. Input rank of a node: its parent's rank bound
# plus its own low-rank terms.
function _nodeinputrank(P, node)
    total = node.parent == 0 ? 0 : P.nodes[node.parent].estimate
    for panel in node.afar
        total += rank(P.a.workspace.far[P.a.workspace.csr.storageidx[panel.entry]])
    end
    for panel in node.bfar
        total += rank(P.b.workspace.far[P.b.workspace.csr.storageidx[panel.entry]])
    end
    return total
end

# Pending capacity, dense-core choice and buffers of destination `g`; reads the
# rank bound of its nearest node, which must already have run.
function _preparegroup!(P, g, scratch)
    C = P.destination.workspace
    if !C.csr.isfar[g.destination]
        # Dense destinations expand low-rank panels through a thin temporary.
        right = 0
        for panel in g.afar
            r = rank(P.a.workspace.far[P.a.workspace.csr.storageidx[panel.entry]])
            right = max(right, checked_mul(r, length(g.cols)))
        end
        for panel in g.bfar
            r = rank(P.b.workspace.far[P.b.workspace.csr.storageidx[panel.entry]])
            right = max(right, checked_mul(length(g.rows), r))
        end
        _reserveproduct!(scratch, 0, right, 0)
        return nothing
    end
    m, n = size(C.far[C.csr.storageidx[g.destination]])
    estimated = _productrankestimate(P, g)
    g.capacity = estimated >= P.batchrank ? P.batchrank : floor(Int, estimated)
    left, right, aggregate = checked_mul(m, g.capacity), checked_mul(n, g.capacity), 0
    if g.implicit
        # Sketch intermediates: B*Ω / Q'*A over the contracted axis, Q'.
        width = _implicitwidth(P, m, n)
        aggregate = checked_mul(length(P.a.cols), width)
        left = max(left, checked_mul(m, width))
    end
    # Dense/dense pairs contribute factor slices as wide as their inner range;
    # beyond mn/(m+n) a dense m × n core is cheaper. Low-rank panels stay
    # factored, pending behind the core in the left buffer.
    mn = checked_mul(m, n)
    g.dense = g.denserank * (m + n) >= mn
    if g.dense
        lowrank = round(Int, estimated - g.denserank)
        g.capacity = max(1, min(lowrank, max(P.batchrank, mn ÷ (m + n))))
        left = max(left, checked_add(mn, checked_mul(m, g.capacity)))
        right = max(right, checked_mul(n, g.capacity))
    end
    _reserveproduct!(scratch, left, right, aggregate)
    return nothing
end

@inline function _productpending(P, g, r, scratch)
    W, csr = P.destination.workspace, P.destination.workspace.csr
    m, n = size(W.far[csr.storageidx[g.destination]])
    offset = g.dense ? m * n : 0
    U = reshape(
        view(scratch.productleft, (offset + 1):(offset + m * g.capacity)), m, g.capacity
    )
    V = reshape(view(scratch.productright, 1:(n * g.capacity)), g.capacity, n)
    return view(U, :, 1:r), view(V, 1:r, :)
end

# Physical dense core of a far destination, in front of its pending factors.
@inline function _densecore(P, g, scratch)
    W, csr = P.destination.workspace, P.destination.workspace.csr
    m, n = size(W.far[csr.storageidx[g.destination]])
    return reshape(view(scratch.productleft, 1:(m * n)), m, n)
end

function _flushproduct!(P, g, pending, scratch, status, options)
    pending == 0 && return nothing
    W, csr = P.destination.workspace, P.destination.workspace.csr
    R = W.far[csr.storageidx[g.destination]]
    U, V = _productpending(P, g, pending, scratch)
    if g.dense
        # Full pending storage beside a dense core: absorb it with one GEMM.
        mul!(_densecore(P, g, scratch), U, V, true, true)
        return nothing
    end
    _record!(status, _flushfactors!(R, U, V, scratch; options..., sketch=true))
    return nothing
end

# Dense A terminal i times dense B terminal j over the inner range `inner`.
function _productpair!(
    P, g, i, j, inner, alpha, scratch, pending, status, options, denseoutput
)
    WA, WB = P.a.workspace, P.b.workspace
    rows = _overlap(g.rows, _rowrange(WA, i))
    cols = _overlap(g.cols, _colrange(WB, j))
    A = view(
        WA.near[WA.csr.storageidx[i]],
        _relative(rows, _rowrange(WA, i)),
        _relative(inner, _colrange(WA, i)),
    )
    B = view(
        WB.near[WB.csr.storageidx[j]],
        _relative(inner, _rowrange(WB, j)),
        _relative(cols, _colrange(WB, j)),
    )
    return _productpairblocks!(
        P, g, A, B, rows, cols, alpha, scratch, pending, status, options, denseoutput
    )
end

function _productpairblocks!(
    P, g, A, B, rows, cols, alpha, scratch, pending, status, options, denseoutput
)
    W, csr = P.destination.workspace, P.destination.workspace.csr
    k = g.destination
    rr, cc = _productranges(W, k)
    rrows, rcols = _relative(rows, rr), _relative(cols, cc)
    if denseoutput !== nothing
        mul!(view(denseoutput, rrows, rcols), A, B, alpha, true)
        return pending
    end
    total = size(A, 2)
    offset = 0
    while offset < total
        if pending == g.capacity
            _flushproduct!(P, g, pending, scratch, status, options)
            pending = 0
        end
        r = min(total - offset, g.capacity - pending)
        U, V = _productpending(P, g, pending + r, scratch)
        up, vp = view(U, :, (pending + 1):(pending + r)),
        view(V, (pending + 1):(pending + r), :)
        fill!(up, zero(eltype(up)))
        fill!(vp, zero(eltype(vp)))
        columns = (offset + 1):(offset + r)
        X, Y = view(up, rrows, :), view(vp, :, rcols)
        copyto!(X, view(A, :, columns))
        copyto!(Y, view(B, columns, :))
        size(X, 1) <= size(Y, 2) ? rmul!(X, alpha) : lmul!(alpha, Y)
        pending += r
        offset += r
    end
    return pending
end

function _joinproduct!(
    P,
    g,
    i,
    interval,
    node,
    lo,
    hi,
    alpha,
    scratch,
    pending,
    status,
    options,
    denseoutput=nothing,
)
    lo > hi && return pending
    g.bmaxend[node] < first(interval) && return pending
    br, _ = _productranges(P.b.workspace, g.bentries[lo])
    first(br) > last(interval) && return pending
    if lo == hi
        j = g.bentries[lo]
        P.b.workspace.csr.isfar[j] && return pending
        inner = _overlap(interval, br)
        isempty(inner) && return pending
        return _productpair!(
            P, g, i, j, inner, alpha, scratch, pending, status, options, denseoutput
        )
    end
    mid = (lo + hi) >>> 1
    pending = _joinproduct!(
        P,
        g,
        i,
        interval,
        2node,
        lo,
        mid,
        alpha,
        scratch,
        pending,
        status,
        options,
        denseoutput,
    )
    return _joinproduct!(
        P,
        g,
        i,
        interval,
        2node + 1,
        mid + 1,
        hi,
        alpha,
        scratch,
        pending,
        status,
        options,
        denseoutput,
    )
end

"""
    _executenode!(P::_ProductPlan, n, α, scratch, status, options)

Form the low-rank factors of virtual node `n`: the slice of its parent's result
plus the thin products of its own low-rank panels, recompressed once with a
sketch. The result is kept in `P.results[n]` until the destinations below it
have run. The parent node must already have run.
"""
function _executenode!(P::_ProductPlan{T}, n, alpha, scratch, status, options) where {T}
    node = P.nodes[n]
    m, w = length(node.rows), length(node.cols)
    # Rank bound for the destinations below; parents run first.
    node.estimate = min(_nodeinputrank(P, node), m, w)
    inherited = node.parent == 0 ? 0 : rank(P.results[node.parent])
    total, largest = inherited, inherited
    for panel in node.afar
        r = rank(P.a.workspace.far[P.a.workspace.csr.storageidx[panel.entry]])
        total, largest = total + r, max(largest, r)
    end
    for panel in node.bfar
        r = rank(P.b.workspace.far[P.b.workspace.csr.storageidx[panel.entry]])
        total, largest = total + r, max(largest, r)
    end
    _reserveproduct!(scratch, 0, 0, checked_mul(m + w, total))
    U = reshape(view(scratch.aggregate, 1:(m * total)), m, total)
    V = reshape(view(scratch.aggregate, (m * total + 1):((m + w) * total)), total, w)
    if inherited > 0
        parent = P.nodes[node.parent]
        R = P.results[node.parent]
        copyto!(view(U, :, 1:inherited), view(R.U, _relative(node.rows, parent.rows), :))
        copyto!(view(V, 1:inherited, :), view(R.V, :, _relative(node.cols, parent.cols)))
    end
    # Each low-rank term contributes exact factor columns.
    offset = inherited
    for panel in node.afar
        isempty(panel.entries) && continue
        X, Y = _farfactors(P.a, panel.entry, node.rows, panel.rows)
        r = size(X, 2)
        r == 0 && continue
        columns = (offset + 1):(offset + r)
        copyto!(view(U, :, columns), X)
        _thinright!(
            view(V, columns, :), Y, _panelblock(P.b, panel), alpha, zero(alpha), scratch
        )
        offset += r
    end
    for panel in node.bfar
        X, Y = _farfactors(P.b, panel.entry, panel.cols, node.cols)
        r = size(X, 2)
        r == 0 && continue
        columns = (offset + 1):(offset + r)
        _thinleft!(
            view(U, :, columns), _panelblock(P.a, panel), X, alpha, zero(alpha), scratch
        )
        copyto!(view(V, columns, :), Y)
        offset += r
    end
    R = LowRankMatrix{T}(Matrix{T}(undef, m, 0), Matrix{T}(undef, 0, w))
    if offset == inherited
        # Nothing new at this node: pass the parent slice through unchanged.
        R.U, R.V = U[:, 1:offset], V[1:offset, :]
    else
        # A sum of smooth terms rarely exceeds its largest term's rank.
        info = _flushfactors!(
            R,
            view(U, :, 1:offset),
            view(V, 1:offset, :),
            scratch;
            options...,
            sketch=true,
            guess=largest + 3,
        )
        _record!(status, info, 0)
    end
    P.results[n] = R
    return nothing
end

"""
    _ProductOperator

Implicit far destination `C + α*A*B` for [`_sketchrecompression!`](@ref),
applied through hierarchical thin products (`_thinmul!`). Its cost is the
storage of the operands' panels times the sketch width, however many low-rank
terminals contribute. `inner` holds `B*Ω` or `Q'*A`, `adjointq` holds `Q'`.
"""
struct _ProductOperator{T,R,HA,HB,V}
    C::R
    a::HA
    b::HB
    alpha::T
    inner::V
    adjointq::V
    rank::Int
end
_sketchrank(S::_ProductOperator) = S.rank
_implicitwidth(P, m, n) = min(min(m, n) ÷ 3, P.batchrank) + 8

function _sketchapply!(W, S::_ProductOperator{T}, Ω, scratch) where {T}
    b, rc = size(Ω, 2), rank(S.C)
    p = length(S.b.rows)
    if rc > 0
        Z = reshape(view(scratch.small, 1:(rc * b)), rc, b)
        mul!(Z, S.C.V, Ω)
        mul!(W, S.C.U, Z)
    else
        fill!(W, zero(T))
    end
    Z = reshape(view(S.inner, 1:(p * b)), p, b)
    fill!(Z, zero(T))
    _thinexecute!(Z, Ω, S.b, one(T), scratch.small, Val(:N), Val(:left))
    _thinexecute!(W, Z, S.a, S.alpha, scratch.small, Val(:N), Val(:left))
    return W
end

function _sketchproject!(B, S::_ProductOperator{T}, Q, scratch) where {T}
    m, k, rc = size(Q, 1), size(Q, 2), rank(S.C)
    p = length(S.b.rows)
    if rc > 0
        Z = reshape(view(scratch.small, 1:(k * rc)), k, rc)
        mul!(Z, adjoint(Q), S.C.U)
        mul!(B, Z, S.C.V)
    else
        fill!(B, zero(T))
    end
    Qt = reshape(view(S.adjointq, 1:(k * m)), k, m)
    adjoint!(Qt, Q)
    Z = reshape(view(S.inner, 1:(k * p)), k, p)
    fill!(Z, zero(T))
    _thinexecute!(Z, Qt, S.a, one(T), scratch.small, Val(:N), Val(:right))
    _thinexecute!(B, Z, S.b, S.alpha, scratch.small, Val(:N), Val(:right))
    return B
end

# Returns false, leaving the destination unchanged, if the result's rank would
# exceed the sketch capacity; the caller then uses pending factor panels.
function _implicitproduct!(P, g, alpha, scratch, status, options)
    W, csr = P.destination.workspace, P.destination.workspace.csr
    R = W.far[csr.storageidx[g.destination]]
    m, n = size(R)
    A = _subblock(P.a, g.rows, P.a.cols, g.aentries)
    B = _subblock(P.b, P.a.cols, g.cols, g.bentries)
    touched = max(
        rank(R), _maxrank(P.a.workspace, g.aentries), _maxrank(P.b.workspace, g.bentries)
    )
    S = _ProductOperator(R, A, B, alpha, scratch.aggregate, scratch.productleft, touched)
    (; atol, rtol, maxrank) = options
    qcap = _implicitwidth(P, m, n) - 8
    # Updates rarely raise a far block's rank; every sample column costs a full
    # pass over both operands, and the test vectors catch any rank growth.
    result = _sketchrecompression!(S, m, n, rank(R) + 2, qcap, scratch, atol, rtol, maxrank)
    isnothing(result) && return false
    R.U, R.V = result[1], result[2]
    _record!(status, result[3])
    return true
end

# The nearest node's slice: one low-rank term carrying every ancestor A/B term.
function _deliverproductnode!(P, g, scratch, pending, status, options, denseoutput)
    g.node == 0 && return pending
    node, R = P.nodes[g.node], P.results[g.node]
    rank(R) == 0 && return pending
    U = view(R.U, _relative(g.rows, node.rows), :)
    V = view(R.V, :, _relative(g.cols, node.cols))
    return _productpairblocks!(
        P,
        g,
        U,
        V,
        g.rows,
        g.cols,
        one(eltype(U)),
        scratch,
        pending,
        status,
        options,
        denseoutput,
    )
end

# All updates of a dense destination D, applied directly with BLAS.
function _denseproductupdates!(D, P, g, alpha, scratch, status, options)
    _deliverproductnode!(P, g, scratch, 0, status, options, D)
    rr, cc = _productranges(P.destination.workspace, g.destination)
    # A far terminal times a whole panel: one thin product and one expansion.
    for panel in g.afar
        rows = _overlap(_rowrange(P.a.workspace, panel.entry), g.rows)
        X, Y = _farfactors(P.a, panel.entry, rows, panel.rows)
        r = size(X, 2)
        r == 0 && continue
        K = reshape(view(scratch.productright, 1:(r * length(g.cols))), r, length(g.cols))
        _thinright!(K, Y, _panelblock(P.b, panel), alpha, zero(alpha), scratch)
        mul!(view(D, _relative(rows, rr), _relative(g.cols, cc)), X, K, true, true)
    end
    for panel in g.bfar
        cols = _overlap(_colrange(P.b.workspace, panel.entry), g.cols)
        X, Y = _farfactors(P.b, panel.entry, panel.cols, cols)
        r = size(X, 2)
        r == 0 && continue
        K = reshape(view(scratch.productright, 1:(length(g.rows) * r)), length(g.rows), r)
        _thinleft!(K, _panelblock(P.a, panel), X, alpha, zero(alpha), scratch)
        mul!(view(D, _relative(g.rows, rr), _relative(cols, cc)), K, Y, true, true)
    end
    _joindense!(P, g, alpha, scratch, 0, status, options, D)
    return nothing
end

function _executeproduct!(P, alpha, scratch, status, options)
    resize!(P.results, length(P.nodes))
    for n in eachindex(P.nodes)
        _executenode!(P, n, alpha, scratch, status, options)
    end
    for g in P.groups
        _executegroup!(P, g, alpha, scratch, status, options)
    end
    # Node factors are transient; do not retain them between executions.
    empty!(P.results)
    return nothing
end

"""
    _executegroup!(P::_ProductPlan, g::_ProductGroup, α, scratch, status, options)

Apply all updates of destination `g`. A dense destination is updated directly
with BLAS. A far destination is recompressed from the implicit operator if
planned so; otherwise its low-rank panel products, the slice of its nearest
node and its dense/dense pairs are collected as pending factor columns (beside
a dense core if chosen) and recompressed together with the destination, in
batches of at most `capacity` columns. Its nearest node must already have run.
"""
function _executegroup!(P, g, alpha, scratch, status, options)
    _preparegroup!(P, g, scratch)
    csr = P.destination.workspace.csr
    if !csr.isfar[g.destination]
        D = P.destination.workspace.near[csr.storageidx[g.destination]]
        _denseproductupdates!(D, P, g, alpha, scratch, status, options)
        return nothing
    end
    g.implicit && _implicitproduct!(P, g, alpha, scratch, status, options) && return nothing
    g.dense && fill!(_densecore(P, g, scratch), zero(eltype(scratch.productleft)))
    # A far terminal times a panel is one low-rank update. Against a single far
    # terminal of lower rank, that terminal's factors are carried instead.
    pending = 0
    for panel in g.afar
        rows = _overlap(_rowrange(P.a.workspace, panel.entry), g.rows)
        X, Y = _farfactors(P.a, panel.entry, rows, panel.rows)
        if _singlefar(P.b, panel)
            j = only(panel.entries)
            BX, BY = _farfactors(P.b, j, panel.rows, panel.cols)
            if size(BX, 2) < size(X, 2)
                H = _subblock(P.a, rows, panel.rows, (panel.entry):(panel.entry))
                pending = _productpanelchunks!(
                    P,
                    g,
                    BX,
                    BY,
                    H,
                    rows,
                    panel.cols,
                    alpha,
                    scratch,
                    pending,
                    status,
                    options,
                    Val(:right),
                )
                continue
            end
        end
        pending = _productpanelchunks!(
            P,
            g,
            X,
            Y,
            _panelblock(P.b, panel),
            rows,
            g.cols,
            alpha,
            scratch,
            pending,
            status,
            options,
            Val(:left),
        )
    end
    for panel in g.bfar
        cols = _overlap(_colrange(P.b.workspace, panel.entry), g.cols)
        X, Y = _farfactors(P.b, panel.entry, panel.cols, cols)
        pending = _productpanelchunks!(
            P,
            g,
            X,
            Y,
            _panelblock(P.a, panel),
            g.rows,
            cols,
            alpha,
            scratch,
            pending,
            status,
            options,
            Val(:right),
        )
    end
    pending = _deliverproductnode!(P, g, scratch, pending, status, options, nothing)
    if !g.dense
        pending = _joindense!(P, g, alpha, scratch, pending, status, options, nothing)
        _flushproduct!(P, g, pending, scratch, status, options)
    else
        D = _densecore(P, g, scratch)
        pending = _joindense!(P, g, alpha, scratch, pending, status, options, D)
        _finishdenseproduct!(P, g, pending, scratch, status, options)
    end
    return nothing
end

# Dense/dense terminal pairs: factor slices for a far destination, or direct
# updates of `denseoutput` (a dense core or destination).
function _joindense!(P, g, alpha, scratch, pending, status, options, denseoutput)
    interval = P.a.cols
    for i in g.aentries
        P.a.workspace.csr.isfar[i] && continue
        _, ac = _productranges(P.a.workspace, i)
        inner = _overlap(ac, interval)
        pending = _joinproduct!(
            P,
            g,
            i,
            inner,
            1,
            1,
            length(g.bentries),
            alpha,
            scratch,
            pending,
            status,
            options,
            denseoutput,
        )
    end
    return pending
end

# Recompress dense core + destination + pending low-rank factors with one sketch,
# without expanding the factored terms. The dense SVD remains the fallback.
function _finishdenseproduct!(P, g, pending, scratch, status, options)
    W, csr = P.destination.workspace, P.destination.workspace.csr
    R = W.far[csr.storageidx[g.destination]]
    m, n = size(R)
    D = _densecore(P, g, scratch)
    U, V = _productpending(P, g, pending, scratch)
    (; atol, rtol, maxrank) = options
    S = _FactoredSum(D, R.U, R.V, U, V)
    result = _sketchrecompression!(
        S, m, n, rank(R) + 5, min(m, n) ÷ 3, scratch, atol, rtol, maxrank
    )
    if isnothing(result)
        mul!(D, R.U, R.V, true, true)
        mul!(D, U, V, true, true)
        buffers = _recompressionbuffers!(scratch, m, n, max(m, n))
        copyto!(buffers.small, D)
        result = _finishrecompression(buffers, m, n, max(m, n), atol, rtol, maxrank)
    end
    R.U, R.V = result[1], result[2]
    _record!(status, result[3])
    return nothing
end

# Destinations follow their nearest node, nodes their parent node; task t is
# node t, then destination t - length(P.nodes).
function _productgraph(P)
    builder = _TaskGraphBuilder(0)
    for node in P.nodes
        _addtask!(builder, (), (), node.parent == 0 ? () : (node.parent,))
    end
    for g in P.groups
        _addtask!(builder, (), (), g.node == 0 ? () : (g.node,))
    end
    return _TaskGraph(builder)
end

"""
    _addmul!(P::_ProductPlan, α, workers, graph::_TaskGraph, options)

`C += α*A*B` with the blocks of the plan, distributed over `workers` along
`graph` (from `_productgraph(P)`). Every touched far block is recompressed to
`options`. Returns the aggregated truncation status.
"""
function _addmul!(
    P::_ProductPlan{T}, alpha::Number, workers::AbstractVector, graph::_TaskGraph, options
) where {T}
    alpha = convert(T, alpha)
    isfinite(alpha) || throw(ArgumentError("alpha must be finite"))
    _resetstatus!(workers)
    iszero(alpha) ||
        isempty(P.a.cols) ||
        isempty(P.destination.rows) ||
        isempty(P.destination.cols) ||
        _runproduct!(P, alpha, workers, graph, options)
    return _workerstatus(workers)
end

function _runproduct!(P, alpha, workers, graph, options)
    nodes = length(P.nodes)
    resize!(P.results, nodes)
    try
        _rungraph!(graph, workers) do t, worker
            scratch, status = worker.scratch, worker.status
            if t <= nodes
                _executenode!(P, t, alpha, scratch, status, options)
            else
                _executegroup!(P, P.groups[t - nodes], alpha, scratch, status, options)
            end
            return true
        end
    finally
        # Node factors are transient.
        empty!(P.results)
    end
    return nothing
end
