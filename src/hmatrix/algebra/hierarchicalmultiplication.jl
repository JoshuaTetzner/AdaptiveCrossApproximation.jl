# Structural execution indices only: source terminal partitions may differ.
# The interval tree joins contracted-axis ranges without an all-pairs scan.
struct _ProductPanel
    entry::Int
    rows::UnitRange{Int}
    cols::UnitRange{Int}
    entries::Vector{Int}
end

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

# Virtual block (row cluster × column cluster) above several destinations. A
# low-rank A terminal whose rows strictly contain a destination's rows reaches
# every destination below the node with the same rows; its thin product is
# formed once per node, summed with the B-far terms and the parent node's slice,
# and recompressed once. Destinations then receive one slice of the nearest node.
mutable struct _ProductNode
    rows::UnitRange{Int}
    cols::UnitRange{Int}
    parent::Int
    afar::Vector{_ProductPanel}
    bfar::Vector{_ProductPanel}
    estimate::Int
end

# Reusable execution plan for C += alpha*A*B on hierarchical blocks.
# Corresponding axes use the same cluster trees and tree-order ranges; terminal
# partitions may differ. Numerical ranks may change between executions, the
# partitions may not. Numerical buffers belong to the executing task's
# BlockScratch, so one plan's nodes and destinations can run on different tasks.
# At most `batchrank` pending update columns are kept per far destination.
#
# Planning shares work between destinations in two ways. A low-rank source
# terminal whose rows (or columns) strictly contain several complete
# destinations is attached to a virtual node (row cluster x column cluster)
# above them; each node forms its thin products once, adds its parent's slice,
# is recompressed once, and destinations receive one slice of their nearest
# node. A complete far destination for which it is estimated cheaper is
# recompressed directly from the implicit operator C + alpha*A*B through
# hierarchical thin products, with the factor panels as fallback. A far
# destination whose dense/dense contributions reach rank m*n/(m+n) sums them
# into a dense core instead.
mutable struct _ProductPlan{T,C,A,B}
    destination::C
    a::A
    b::B
    groups::Vector{_ProductGroup}
    nodes::Vector{_ProductNode}
    results::Vector{LowRankMatrix{T}}
    batchrank::Int
    spans::Vector{Tuple{UInt,UInt,Bool}}
end

_productaxes(H) = (
    testtree(H.workspace.tree),
    trialtree(H.workspace.tree),
    H.workspace.rowperm,
    H.workspace.colperm,
    H.rows,
    H.cols,
)

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

# Cluster IDs of a source terminal.
_productnodes(W, k) = (W.entryrows[k], W.csr.colnodes[k])

function _clusterparents(tree, count)
    parents = zeros(Int, count)
    stack = Int[_clusterroot(tree)]
    while !isempty(stack)
        node = pop!(stack)
        for child in _clusterchildren(tree, node)
            parents[child] = node
            push!(stack, child)
        end
    end
    return parents
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

# Plan-time flop model for one complete far destination with current ranks.
# Implicit: every sketch column passes once through B(inner, cols) and A(rows,
# inner). Factors: each local low-rank term (ancestor terms are shared through
# nodes) costs a thin product plus its pending columns in the sketch, and dense
# pairs enter as factor slices or a dense core.
function _implicitcheaper(C, A, B, k, rows, cols, ai, bi, denserank, parents)
    W = C.workspace
    m, n = length(rows), length(cols)
    width = 2 * rank(W.far[W.csr.storageidx[k]]) + 18
    inner = _productaxes(A)[6]
    implicit =
        width * (_panelstorage(A, ai, rows, inner) + _panelstorage(B, bi, inner, cols))
    rowparents, colparents = parents
    τ, σ = W.entryrows[k], W.csr.colnodes[k]
    factors = min(denserank, m * n / (m + n)) * (m + n) * width
    for i in ai
        A.workspace.csr.isfar[i] || continue
        _ancestorsteps(rowparents, τ, first(_productnodes(A.workspace, i))) == 0 || continue
        r = rank(A.workspace.far[A.workspace.csr.storageidx[i]])
        _, ac = _productranges(A.workspace, i)
        interval = _overlap(ac, inner)
        entries = _productsubentries(B, interval, cols)
        factors += r * (_panelstorage(B, entries, interval, cols) + (m + n) * width)
        factors > implicit && return true
    end
    for j in bi
        B.workspace.csr.isfar[j] || continue
        _ancestorsteps(colparents, σ, last(_productnodes(B.workspace, j))) == 0 || continue
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

function _productgroups(C, A, B)
    groups = _ProductGroup[]
    W = C.workspace
    rowtree, coltree = _productaxes(C)[1:2]
    rowparents = _clusterparents(rowtree, length(W.rowranges))
    colparents =
        coltree === rowtree ? rowparents : _clusterparents(coltree, length(W.colranges))
    inner = _productaxes(A)[6]
    # Complete far destinations may be recompressed as the
    # implicit operator C + alpha*A*B. Only complete destinations receive node
    # slices; clipped ones keep panels.
    whole, implicit = Bool[], Bool[]
    for k in C.entries
        rr, cc = _productranges(W, k)
        rows, cols = _overlap(rr, C.rows), _overlap(cc, C.cols)
        ai = _productsubentries(A, rows, inner)
        bi = _productsubentries(B, _productaxes(B)[5], cols)
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
            key = _productkey(
                rowparents, colparents, τ, σ, first(_productnodes(A.workspace, i)), 0
            )
            isnothing(key) || (candidates[key] = 0)
        end
        for j in g.bentries
            B.workspace.csr.isfar[j] || continue
            key = _productkey(
                rowparents, colparents, τ, σ, 0, last(_productnodes(B.workspace, j))
            )
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
        push!(depths, _ancestorsteps(rowparents, key[1], _clusterroot(rowtree)))
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
        # Remaining low-rank terminals are consolidated through whole panels.
        # Far destinations append them as pending factors; dense destinations
        # expand each panel once instead of once per terminal pair.
        for i in g.aentries
            A.workspace.csr.isfar[i] || continue
            _, ac = _productranges(A.workspace, i)
            interval = _overlap(ac, inner)
            key = if complete && !isempty(nodes)
                _productkey(
                    rowparents,
                    colparents,
                    τ,
                    σ,
                    first(_productnodes(A.workspace, i)),
                    0,
                )
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
                _productkey(
                    rowparents, colparents, τ, σ, 0, last(_productnodes(B.workspace, j))
                )
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

function _singleproductblock(H, k, rows, cols)
    return HMatrixBlock{eltype(H),typeof(H.workspace),UnitRange{Int}}(
        H.workspace, rows, cols, k:k
    )
end

function _productsmallright!(
    B, P, g, H, rows, cols, alpha, scratch, pending, status, options
)
    X, Y = _productfactors(B)
    return _productpanelchunks!(
        P, g, X, Y, H, rows, cols, alpha, scratch, pending, status, options, Val(:right)
    )
end

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

function _panelblock(H, panel)
    return HMatrixBlock{eltype(H),typeof(H.workspace),Vector{Int}}(
        H.workspace, panel.rows, panel.cols, panel.entries
    )
end

# A far terminal times a hierarchical panel is a single low-rank update.
# For B-far panels the hierarchical A panel contains only dense terminals,
# preventing double counting the far-A contributions already handled above.
function _productfarleft!(A, P, g, panel, rows, alpha, scratch, pending, status, options)
    X, Y = _productfactors(A)
    other = _panelblock(P.b, panel)
    return _productpanelchunks!(
        P,
        g,
        X,
        Y,
        other,
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

function _productfarright!(B, P, g, panel, cols, alpha, scratch, pending, status, options)
    X, Y = _productfactors(B)
    other = _panelblock(P.a, panel)
    return _productpanelchunks!(
        P,
        g,
        X,
        Y,
        other,
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
    # Solve updates often address a panel with only a few owners in a large
    # workspace. Scanning that cached panel avoids revisiting the global tree
    # and its much wider CSR/CSC rows for every destination and factor panel.
    if length(H.entries) <= 64
        entries = Int[]
        for k in H.entries
            rr, cc = _productranges(H.workspace, k)
            isempty(_overlap(rr, rows)) || isempty(_overlap(cc, cols)) || push!(entries, k)
        end
        return entries
    end
    selected = blockview(H.workspace, rows, cols)
    # A subview already owns its entry vector; do not duplicate it during
    # construction. Only a full-workspace OneTo needs materializing.
    return if selected.entries isa Vector{Int}
        selected.entries
    else
        collect(Int, selected.entries)
    end
end

function _ProductPlan(
    c::HMatrixBlock{T}, a::HMatrixBlock{T}, b::HMatrixBlock{T}; batchrank::Integer=128
) where {T<:LinearAlgebra.BlasFloat}
    batchrank > 0 || throw(ArgumentError("batchrank must be positive"))
    ca, aa, ba = _productaxes(c), _productaxes(a), _productaxes(b)
    ca[1] === aa[1] && ca[2] === ba[2] && aa[2] === ba[1] ||
        throw(ArgumentError("product axes require matching cluster-tree objects"))
    ca[3] == aa[3] && ca[4] == ba[4] && aa[4] == ba[3] ||
        throw(ArgumentError("product axes have different external orderings"))
    ca[5] == aa[5] && ca[6] == ba[6] && aa[6] == ba[5] ||
        throw(DimensionMismatch("product axes require matching tree-order ranges"))
    groups, nodes = _productgroups(c, a, b)
    return _ProductPlan{T,typeof(c),typeof(a),typeof(b)}(
        c, a, b, groups, nodes, LowRankMatrix{T}[], Int(batchrank), Tuple{UInt,UInt,Bool}[]
    )
end

function _checkproductblockbuffers(block, scratch)
    # Tuple foreach specializes every element. Iterating the heterogeneous
    # real/complex/integer LAPACK tuple boxes iteration states in this hot path.
    foreach(buffer -> _checkproductalias(buffer, block), _scratcharrays(scratch))
    return nothing
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

function _pushproductspan!(spans, array, destination)
    isempty(array) || push!(
        spans, (UInt(pointer(array)), UInt(pointer(array)) + sizeof(array), destination)
    )
    return nothing
end

function _checkproductworkspace(P, scratch)
    P.batchrank > 0 || throw(ArgumentError("batchrank must be positive"))
    spans = P.spans
    empty!(spans)
    GC.@preserve P begin
        for (H, destination) in ((P.destination, true), (P.a, false), (P.b, false)),
            k in H.entries

            W, csr = H.workspace, H.workspace.csr
            _checkworkspaceblock(W, k)
            block = csr.isfar[k] ? W.far[csr.storageidx[k]] : W.near[csr.storageidx[k]]
            if block isa LowRankMatrix
                # Empty factors still share mutable block identity: replacing
                # destination fields must not change a rank-zero source.
                address = UInt(pointer_from_objref(block))
                push!(spans, (address, address + 1, destination))
                _pushproductspan!(spans, block.U, destination)
                _pushproductspan!(spans, block.V, destination)
            else
                _pushproductspan!(spans, block, destination)
            end
            _checkproductblockbuffers(block, scratch)
        end
        sort!(spans; alg=QuickSort)
        destend, sourceend = zero(UInt), zero(UInt)
        for (start, stop, destination) in spans
            if destination
                start >= max(destend, sourceend) || throw(
                    ArgumentError(
                        "product destination overlaps source or another destination"
                    ),
                )
                destend = max(destend, stop)
            else
                start >= destend ||
                    throw(ArgumentError("product source overlaps destination"))
                sourceend = max(sourceend, stop)
            end
        end
    end
    return nothing
end

# Numerical ranks may change between executions. Each node and destination
# derives its rank bounds and buffer sizes from the current ranks of its own
# operands just before it runs; no buffer views survive between them.
# Input rank of a node: its parent's rank bound plus its own low-rank terms.
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
        aggregate = checked_mul(length(_productaxes(P.a)[6]), width)
        left = max(left, checked_mul(m, width))
    end
    # Dense/dense terminal pairs contribute factor slices as wide as their inner
    # range. Once these reach mn/(m+n), summing them into a dense m × n core is
    # cheaper than carrying the slices. Low-rank panels stay factored, pending
    # behind the core in the left buffer.
    mn = checked_mul(m, n)
    g.dense = g.denserank * (m + n) >= mn
    if g.dense
        lowrank = round(Int, estimated - g.denserank)
        g.capacity = max(1, min(lowrank, max(P.batchrank, mn ÷ (m + n))))
        left = max(left, Base.Checked.checked_add(mn, checked_mul(m, g.capacity)))
        right = max(right, checked_mul(n, g.capacity))
    end
    _reserveproduct!(scratch, left, right, aggregate)
    return nothing
end

# Return borrowed numerical views. The caller immediately enters a concrete
# numerical function barrier; no matrix/factor payload is allocated here.
function _withproductblock(f::F, H, k, rows, cols, args::Vararg{Any,N}) where {F,N}
    W, csr = H.workspace, H.workspace.csr
    rr, cc = _productranges(W, k)
    r, c = _relative(rows, rr), _relative(cols, cc)
    if csr.isfar[k]
        R = W.far[csr.storageidx[k]]
        return f(_LowRankSlice(view(R.U, r, :), view(R.V, :, c)), args...)
    end
    return f(view(W.near[csr.storageidx[k]], r, c), args...)
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

# Only dense/dense pairs and node slices reach factor accumulation: low-rank
# terminals have already been consolidated through panels or nodes.
function _productchunk!(U, V, A::AbstractMatrix, B::AbstractMatrix, columns, scratch)
    copyto!(U, view(A, :, columns))
    copyto!(V, view(B, columns, :))
    return nothing
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
    info = _flushfactors!(R, U, V, scratch; options..., sketch=true)
    status.converged &= info.converged
    status.rank_limited |= info.rank_limited
    status.max_block_error = max(status.max_block_error, info.error)
    status.flushes += 1
    return nothing
end

function _productpair!(
    P, g, i, j, inner, alpha, scratch, pending, status, options, denseoutput
)
    ar, _ = _productranges(P.a.workspace, i)
    _, bc = _productranges(P.b.workspace, j)
    rows, cols = _overlap(g.rows, ar), _overlap(g.cols, bc)
    return _withproductblock(
        _productwitha!,
        P.a,
        i,
        rows,
        inner,
        P,
        g,
        j,
        rows,
        cols,
        inner,
        alpha,
        scratch,
        pending,
        status,
        options,
        denseoutput,
    )
end

function _productwitha!(
    A, P, g, j, rows, cols, inner, alpha, scratch, pending, status, options, denseoutput
)
    return _withproductblock(
        _productwithb!,
        P.b,
        j,
        inner,
        cols,
        A,
        P,
        g,
        rows,
        cols,
        alpha,
        scratch,
        pending,
        status,
        options,
        denseoutput,
    )
end

function _productwithb!(
    B, A, P, g, rows, cols, alpha, scratch, pending, status, options, denseoutput
)
    return _productpairblocks!(
        P, g, A, B, rows, cols, alpha, scratch, pending, status, options, denseoutput
    )
end

function _scaledproductchunk!(U, V, A, B, columns, alpha, scratch)
    _productchunk!(U, V, A, B, columns, scratch)
    size(U, 1) <= size(V, 2) ? rmul!(U, alpha) : lmul!(alpha, V)
    return nothing
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
        _scaledproductchunk!(
            view(up, rrows, :),
            view(vp, :, rcols),
            A,
            B,
            (offset + 1):(offset + r),
            alpha,
            scratch,
        )
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
        # Low-rank B terminals are consolidated through panels.
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

# Far A terminal times a whole B panel into a dense destination: one thin product
# T = alpha*V_A*B_panel and one expansion U_A*T. The B-far case mirrors it.
function _densefarleft!(A, P, g, panel, rows, alpha, scratch, D)
    X, Y = _productfactors(A)
    r = size(X, 2)
    r == 0 && return nothing
    rr, cc = _productranges(P.destination.workspace, g.destination)
    K = reshape(view(scratch.productright, 1:(r * length(g.cols))), r, length(g.cols))
    _thinright!(K, Y, _panelblock(P.b, panel), alpha, zero(alpha), scratch)
    mul!(view(D, _relative(rows, rr), _relative(g.cols, cc)), X, K, true, true)
    return nothing
end

function _densefarright!(B, P, g, panel, cols, alpha, scratch, D)
    X, Y = _productfactors(B)
    r = size(X, 2)
    r == 0 && return nothing
    rr, cc = _productranges(P.destination.workspace, g.destination)
    K = reshape(view(scratch.productright, 1:(length(g.rows) * r)), length(g.rows), r)
    _thinleft!(K, _panelblock(P.a, panel), X, alpha, zero(alpha), scratch)
    mul!(view(D, _relative(g.rows, rr), _relative(cols, cc)), K, Y, true, true)
    return nothing
end

# Node terms: a far A terminal against the whole B panel below the node, or the
# dense A panel against a far B terminal. Both write exact factor columns.
function _nodefarleft!(A, P, panel, U, V, offset, alpha, scratch)
    X, Y = _productfactors(A)
    r = size(X, 2)
    r == 0 && return offset
    columns = (offset + 1):(offset + r)
    copyto!(view(U, :, columns), X)
    _thinright!(
        view(V, columns, :), Y, _panelblock(P.b, panel), alpha, zero(alpha), scratch
    )
    return offset + r
end

function _nodefarright!(B, P, panel, U, V, offset, alpha, scratch)
    X, Y = _productfactors(B)
    r = size(X, 2)
    r == 0 && return offset
    columns = (offset + 1):(offset + r)
    _thinleft!(view(U, :, columns), _panelblock(P.a, panel), X, alpha, zero(alpha), scratch)
    copyto!(view(V, columns, :), Y)
    return offset + r
end

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
    offset = inherited
    for panel in node.afar
        isempty(panel.entries) && continue
        offset = _withproductblock(
            _nodefarleft!,
            P.a,
            panel.entry,
            node.rows,
            panel.rows,
            P,
            panel,
            U,
            V,
            offset,
            alpha,
            scratch,
        )
    end
    for panel in node.bfar
        offset = _withproductblock(
            _nodefarright!,
            P.b,
            panel.entry,
            panel.cols,
            node.cols,
            P,
            panel,
            U,
            V,
            offset,
            alpha,
            scratch,
        )
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
        status.converged &= info.converged
        status.rank_limited |= info.rank_limited
        status.max_block_error = max(status.max_block_error, info.error)
    end
    P.results[n] = R
    return nothing
end

# Implicit far destination C + alpha*A*B, sampled through hierarchical
# thin products: the cost scales with the operands' storage times the sketch
# width, independently of how many low-rank terminals contribute. `inner` holds
# B*Ω or Q'*A, `adjointq` holds Q'; thin kernels borrow `scratch.small`, which
# the range finder sizes from `_sketchrank`.
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

function _touchedrank(H, entries)
    r = 0
    for k in entries
        H.workspace.csr.isfar[k] &&
            (r = max(r, rank(H.workspace.far[H.workspace.csr.storageidx[k]])))
    end
    return r
end

# Returns false, leaving the destination unchanged, if the result's rank would
# exceed the sketch capacity; the caller then uses pending factor panels.
function _implicitproduct!(P, g, alpha, scratch, status, options)
    W, csr = P.destination.workspace, P.destination.workspace.csr
    R = W.far[csr.storageidx[g.destination]]
    m, n = size(R)
    inner = _productaxes(P.a)[6]
    A = _panelblock(P.a, _ProductPanel(0, g.rows, inner, g.aentries))
    B = _panelblock(P.b, _ProductPanel(0, inner, g.cols, g.bentries))
    touched = max(rank(R), _touchedrank(P.a, g.aentries), _touchedrank(P.b, g.bentries))
    S = _ProductOperator(R, A, B, alpha, scratch.aggregate, scratch.productleft, touched)
    (; atol, rtol, maxrank) = options
    qcap = _implicitwidth(P, m, n) - 8
    # Updates rarely raise a far block's rank; every sample column costs a full
    # pass over both operands, and the test vectors catch any rank growth.
    result = _sketchrecompression!(S, m, n, rank(R) + 2, qcap, scratch, atol, rtol, maxrank)
    isnothing(result) && return false
    R.U, R.V = result[1], result[2]
    info = result[3]
    status.converged &= info.converged
    status.rank_limited |= info.rank_limited
    status.max_block_error = max(status.max_block_error, info.error)
    status.flushes += 1
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

function _denseproductupdates!(D, P, g, alpha, scratch, status, options)
    _deliverproductnode!(P, g, scratch, 0, status, options, D)
    for panel in g.afar
        ar, _ = _productranges(P.a.workspace, panel.entry)
        rows = _overlap(ar, g.rows)
        _withproductblock(
            _densefarleft!,
            P.a,
            panel.entry,
            rows,
            panel.rows,
            P,
            g,
            panel,
            rows,
            alpha,
            scratch,
            D,
        )
    end
    for panel in g.bfar
        _, bc = _productranges(P.b.workspace, panel.entry)
        cols = _overlap(bc, g.cols)
        _withproductblock(
            _densefarright!,
            P.b,
            panel.entry,
            panel.cols,
            cols,
            P,
            g,
            panel,
            cols,
            alpha,
            scratch,
            D,
        )
    end
    interval = _productaxes(P.a)[6]
    for i in g.aentries
        P.a.workspace.csr.isfar[i] && continue
        _, ac = _productranges(P.a.workspace, i)
        inner = _overlap(ac, interval)
        _joinproduct!(
            P, g, i, inner, 1, 1, length(g.bentries), alpha, scratch, 0, status, options, D
        )
    end
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

# All updates of one destination. Its nearest node must already have run.
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
    pending = 0
    for panel in g.afar
        ar, _ = _productranges(P.a.workspace, panel.entry)
        rows = _overlap(ar, g.rows)
        if _singlefar(P.b, panel)
            j = only(panel.entries)
            ra = rank(P.a.workspace.far[P.a.workspace.csr.storageidx[panel.entry]])
            rb = rank(P.b.workspace.far[P.b.workspace.csr.storageidx[j]])
            if rb < ra
                H = _singleproductblock(P.a, panel.entry, rows, panel.rows)
                pending = _withproductblock(
                    _productsmallright!,
                    P.b,
                    j,
                    panel.rows,
                    panel.cols,
                    P,
                    g,
                    H,
                    rows,
                    panel.cols,
                    alpha,
                    scratch,
                    pending,
                    status,
                    options,
                )
                continue
            end
        end
        pending = _withproductblock(
            _productfarleft!,
            P.a,
            panel.entry,
            rows,
            panel.rows,
            P,
            g,
            panel,
            rows,
            alpha,
            scratch,
            pending,
            status,
            options,
        )
    end
    for panel in g.bfar
        _, bc = _productranges(P.b.workspace, panel.entry)
        cols = _overlap(bc, g.cols)
        pending = _withproductblock(
            _productfarright!,
            P.b,
            panel.entry,
            panel.cols,
            cols,
            P,
            g,
            panel,
            cols,
            alpha,
            scratch,
            pending,
            status,
            options,
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
# BLAS updates of `denseoutput` (a dense core). Low-rank terminals use panels.
function _joindense!(P, g, alpha, scratch, pending, status, options, denseoutput)
    interval = _productaxes(P.a)[6]
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
    info = result[3]
    status.converged &= info.converged
    status.rank_limited |= info.rank_limited
    status.max_block_error = max(status.max_block_error, info.error)
    status.flushes += 1
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

# C += alpha*A*B with the plan's blocks, distributed over `workers` along
# `graph`. Returns the aggregated local truncation status. Structural, scalar
# and alias checks precede writes; numerical failures may leave earlier
# destinations updated.
function _addmul!(
    P::_ProductPlan{T}, alpha::Number, workers::AbstractVector, graph::_TaskGraph, options
) where {T}
    alpha = convert(T, alpha)
    isfinite(alpha) || throw(ArgumentError("alpha must be finite"))
    _checkproductworkspace(P, first(workers).scratch)
    _resetstatus!(workers)
    iszero(alpha) ||
        isempty(_productaxes(P.a)[6]) ||
        isempty(P.destination.rows) ||
        isempty(P.destination.cols) ||
        _runproduct!(P, alpha, workers, graph, options)
    return _workerstatus(workers)
end

_addmul!(P::_ProductPlan, alpha::Number, scratch::BlockScratch, options) =
    _addmul!(P, alpha, [_Worker(scratch)], _productgraph(P), options)

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
        # Node factors are transient; do not retain them between executions.
        empty!(P.results)
    end
    return nothing
end
