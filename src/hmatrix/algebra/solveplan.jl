# Reusable execution plan for a triangular solve with a hierarchical RHS block:
# H*X = B (`side=:left`) or X*H = B (`:right`) for an untransposed
# HMatrixTriangular H. The contracted axis uses the factor's cluster tree and
# range; all arithmetic is in tree order.
#
# Each operation owns one complete physical RHS block. It first accumulates all
# external predecessor contributions, then solves the complete diagonal cluster
# through the thin kernels. Low-rank blocks spanning several elimination
# subtrees are solved through their relevant factor, without splitting it.
# Views cutting physical RHS blocks and RHS clusters cutting dense diagonal
# factor leaves are rejected during construction. Ranks and RHS arrays may
# change between executions, the partition may not.
struct _SolvePlan{T,H,B,S,P,R}
    factor::H
    rhs::B
    subfactors::Vector{H}
    entries::Vector{Int}
    submap::Vector{Int}
    updatemap::Vector{Int}
    products::Vector{P}
    dependencyptr::Vector{Int}
    dependencies::Vector{Int}
    spans::Vector{Tuple{UInt,UInt,Bool}}
    status::R
    side::S
end

# Product plans stay concrete even if there are no updates (one root RHS block).
function _solveproducttype(::Type{T}, H, B, ::Val{S}) where {T,S}
    coefficient = HMatrixBlock{T,typeof(H.workspace),Vector{Int}}
    rhs = HMatrixBlock{T,typeof(B.workspace),Vector{Int}}
    return if S === :left
        _ProductPlan{T,rhs,coefficient,rhs}
    else
        _ProductPlan{T,rhs,rhs,coefficient}
    end
end

function _solveblock(W, rows, cols)
    block = blockview(W, rows, cols)
    entries = block.entries isa Vector{Int} ? block.entries : collect(block.entries)
    return HMatrixBlock(W, rows, cols, entries)
end

function _checksolveownership(B)
    W = B.workspace
    for k in B.entries
        rr, cc = _productranges(W, k)
        _overlap(rr, B.rows) == rr && _overlap(cc, B.cols) == cc || throw(
            ArgumentError("hierarchical RHS views must contain complete physical blocks"),
        )
        _checkworkspaceblock(W, k)
    end
    return nothing
end

function _SolvePlan(
    H::HMatrixTriangular{T}, B::HMatrixBlock{T}; side::Symbol=:left, batchrank::Integer=128
) where {T<:LinearAlgebra.BlasFloat}
    _leafside(side)
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
    _checksolveownership(B)
    forward = (H.uplo == 'L') == (S === :left)
    entries = collect(B.entries)
    sortrange(k) = S === :left ? W.rowranges[W.entryrows[k]] : W.colranges[csr.colnodes[k]]
    sort!(entries; by=k -> forward ? first(sortrange(k)) : -last(sortrange(k)))
    # Construction-only dictionaries scale with selected owners, not the full
    # matrix's CSR/tree size. Execution uses only the compact integer vectors.
    positions = Dict{Int,Int}()
    for (i, k) in enumerate(entries)
        positions[k] = i
    end
    subfactors = typeof(H)[]
    submap, updatemap = Int[], Int[]
    nodemap = Dict{Int,Int}()
    products = _solveproducttype(T, H, B, side)[]
    dependencyptr, dependencies = [1], Int[]
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
            for sourceentry in source.entries
                dependency = get(positions, sourceentry, 0)
                0 < dependency < step || throw(
                    ArgumentError(
                        "unsupported RHS partition crossing: predecessor is not a solved owner",
                    ),
                )
                push!(dependencies, dependency)
            end
            rows, cols = S === :left ? (range, previous) : (previous, range)
            coefficient = _solveblock(H.workspace, rows, cols)
            destination = HMatrixBlock(W, rr, cc, [k])
            A, F = S === :left ? (coefficient, source) : (source, coefficient)
            push!(products, _ProductPlan(destination, A, F; batchrank))
            update = length(products)
        end
        push!(updatemap, update)
        push!(dependencyptr, length(dependencies) + 1)
    end
    status = _ProductStatus(true, false, zero(real(T)), 0)
    return _SolvePlan{T,typeof(H),typeof(B),typeof(side),eltype(products),typeof(status)}(
        H,
        B,
        subfactors,
        entries,
        submap,
        updatemap,
        products,
        dependencyptr,
        dependencies,
        Tuple{UInt,UInt,Bool}[],
        status,
        side,
    )
end

function _pushsolveblock!(spans, R::LowRankMatrix, destination, scratch)
    address = UInt(pointer_from_objref(R))
    push!(spans, (address, address + 1, destination))
    _pushproductspan!(spans, R.U, destination)
    _pushproductspan!(spans, R.V, destination)
    destination && _checkproductalias(R.U, R.V)
    isnothing(scratch) || _checkproductblockbuffers(R, scratch)
    return nothing
end
function _pushsolveblock!(spans, D::Matrix, destination, scratch)
    _pushproductspan!(spans, D, destination)
    isnothing(scratch) || _checkproductblockbuffers(D, scratch)
    return nothing
end

# One global preflight, not a full factor/RHS validation for every owner.
function _checksolveplan!(P, scratch; pivots=false)
    H, W, spans = P.factor, P.rhs.workspace, P.spans
    csr = H.workspace.csr
    for i in H.leafspan
        F = H.leaves[i]
        F.factors === H.workspace.near[csr.storageidx[H.leafrecords[i]]] ||
            throw(ArgumentError("diagonal storage was replaced; rebuild the factor plan"))
        _checkleafstate(F)
        pivots && _checkleafpivots(F)
    end
    empty!(spans)
    GC.@preserve P begin
        for (workspace, entries, destination) in
            ((W, P.entries, true), (H.workspace, H.entries, false))
            index = workspace.csr
            for k in entries
                _checkworkspaceblock(workspace, k)
                if index.isfar[k]
                    _pushsolveblock!(
                        spans, workspace.far[index.storageidx[k]], destination, scratch
                    )
                else
                    _pushsolveblock!(
                        spans, workspace.near[index.storageidx[k]], destination, scratch
                    )
                end
            end
        end
        sort!(spans; alg=QuickSort)
        destend, sourceend = zero(UInt), zero(UInt)
        for (start, stop, destination) in spans
            if destination
                start >= max(destend, sourceend) || throw(
                    ArgumentError(
                        "RHS blocks overlap each other or triangular coefficients"
                    ),
                )
                destend = max(destend, stop)
            else
                start >= destend ||
                    throw(ArgumentError("RHS overlaps triangular coefficients"))
                sourceend = max(sourceend, stop)
            end
        end
    end
    return nothing
end

function _solveownerfactor!(B, H, scratch, side::Val{S}) where {S}
    rhs = _leafrhs(B, S === :left ? 'L' : 'R')
    isempty(rhs) && return nothing
    maximumrank = 0
    csr = H.workspace.csr
    for k in H.entries
        csr.isfar[k] &&
            (maximumrank = max(maximumrank, rank(H.workspace.far[csr.storageidx[k]])))
    end
    width = size(rhs, S === :left ? 2 : 1)
    _growscratch!(scratch.small, checked_mul(maximumrank, width))
    _executetriangular!(rhs, H, scratch.small, Val(:N), side)
    return nothing
end

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

# Solve in place with the plan. Returns the aggregated local truncation status
# of the update recompressions. Global alias, shape and leaf-state checks
# precede numerical writes.
function _solve!(P::_SolvePlan{T}, scratch::BlockScratch{T}, options) where {T}
    _checksolveplan!(P, scratch)
    status = P.status
    status.converged, status.rank_limited = true, false
    status.max_block_error, status.flushes = zero(real(T)), 0
    for step in eachindex(P.entries)
        _executesolveoperation!(P, step, scratch, status, options)
    end
    return (
        converged=status.converged,
        rank_limited=status.rank_limited,
        max_block_error=status.max_block_error,
        flushes=status.flushes,
    )
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

# Apply the factor's diagonal-leaf row permutation to the RHS owners of a
# left-sided plan: P*B; `inverse=true` applies P'.
function _pivot!(P::_SolvePlan; inverse::Bool=false)
    P.side === Val(:left) || throw(ArgumentError("pivots apply to left-sided solves"))
    _checksolveplan!(P, nothing; pivots=true)
    return _applypivots!(P, inverse)
end

function _applypivots!(P::_SolvePlan, inverse)
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
