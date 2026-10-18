# Pivot-only panel metadata. Unlike a triangular SolvePlan, this may safely
# address a row slice of an ancestor low-rank owner: permutations act directly
# on the corresponding rows of U and neither change rank nor mix columns.
struct _LUPivotPlan{H,B}
    factor::H
    rhs::B
    leafptr::Vector{Int}
    leafmap::Vector{Int}
end

function _LUPivotPlan(H::HMatrixTriangular, B::HMatrixBlock)
    B.workspace === H.workspace ||
        throw(ArgumentError("LU pivot panels must belong to the factor workspace"))
    B.rows == H.range || throw(DimensionMismatch("LU pivot-panel range differs"))
    W = H.workspace
    leafptr, leafmap = [1], Int[]
    for k in B.entries
        rows, _ = _productranges(W, k)
        for i in H.leafspan
            node = W.entryrows[H.leafrecords[i]]
            leafrange = W.rowranges[node]
            isempty(_overlap(rows, leafrange)) && continue
            _overlap(rows, leafrange) == leafrange || throw(
                ArgumentError(
                    "an LU pivot panel splits dense diagonal leaf $node; refine its storage owner",
                ),
            )
            push!(leafmap, i)
        end
        push!(leafptr, length(leafmap) + 1)
    end
    return _LUPivotPlan(H, B, leafptr, leafmap)
end

function _executelupivots!(P::_LUPivotPlan)
    H, B, W, csr = P.factor, P.rhs, P.rhs.workspace, P.rhs.workspace.csr
    isempty(B.entries) && return B
    for i in H.leafspan
        F = H.leaves[i]
        F.factors === W.near[csr.storageidx[H.leafrecords[i]]] ||
            throw(ArgumentError("diagonal storage was replaced; rebuild the LU plan"))
        _checkleafstate(F)
        _checkleafpivots(F)
    end
    for (position, k) in enumerate(B.entries)
        _checkworkspaceblock(W, k)
        rows, _ = _productranges(W, k)
        active = if csr.isfar[k]
            W.far[csr.storageidx[k]].U
        else
            W.near[csr.storageidx[k]]
        end
        for j in P.leafptr[position]:(P.leafptr[position + 1] - 1)
            i = P.leafmap[j]
            node = W.entryrows[H.leafrecords[i]]
            range = _relative(W.rowranges[node], rows)
            _leafpivots!(view(active, range, :), H.leaves[i].ipiv, false)
        end
    end
    return B
end

struct _LUEliminationStep{P,L,R,M}
    pivots::P
    upper::L
    lower::R
    schur::M
    update::Bool
end

function _luleaves(W::_TreeView{T}) where {T}
    leaves = DenseLeafLU{T,Matrix{T}}[]
    csr = W.csr
    for k in eachindex(csr.storageidx)
        W.entryrows[k] == csr.colnodes[k] || continue
        csr.isfar[k] && throw(ArgumentError("LU diagonal terminals must be dense"))
        A = W.near[csr.storageidx[k]]
        push!(
            leaves,
            DenseLeafLU(
                A,
                Vector{LinearAlgebra.BlasInt}(undef, size(A, 1)),
                LinearAlgebra.BlasInt(-1),
                W.entryrows[k],
            ),
        )
    end
    return leaves
end

function _treeswaps(::IdentityPermutation, n)
    return Int[]
end
function _treeswaps(permutation::AbstractVector{<:Integer}, n)
    length(permutation) == n || throw(DimensionMismatch("invalid tree permutation"))
    order, positions, swaps = collect(1:n), collect(1:n), Vector{Int}(undef, n)
    for i in 1:n
        source = Int(permutation[i])
        1 <= source <= n || throw(ArgumentError("invalid tree permutation"))
        j = positions[source]
        swaps[i] = j
        left, right = order[i], order[j]
        order[i], order[j] = right, left
        positions[left], positions[right] = j, i
    end
    order == permutation || throw(ArgumentError("invalid tree permutation"))
    return swaps
end

function _applytreeswaps!(B, swaps, inverse)
    steps = inverse ? (length(swaps):-1:1) : eachindex(swaps)
    for i in steps
        j = swaps[i]
        i == j && continue
        for column in axes(B, 2)
            B[i, column], B[j, column] = B[j, column], B[i, column]
        end
    end
    return B
end

# Block operations of the factorization, in serial program order.
const _LULEAF, _LUPIVOTS, _LUUPPER, _LULOWER, _LUNODE, _LUGROUP = 1, 2, 3, 4, 5, 6

struct _LUTask
    kind::Int
    step::Int
    item::Int
end

"""
    LUWorkspace(H::HMatrixCore; batchrank=128, ntasks=Threads.nthreads())
    LUWorkspace(workspace::LUWorkspace)

Plans and buffers of the hierarchical LU factorization. The workspace analyses
the partition of `H` once: it builds the elimination schedule, every triangular
solve and Schur-complement product plan, and the dependency graph between their
block operations. It is not bound to `H`: `lu!(A, workspace)` factorizes any
matrix `A` with the same cluster tree and block partition (ranks may differ),
for example the matrices of a frequency sweep on one geometry, without planning
again.

The factorization runs on `ntasks` parallel tasks. Every block operation starts
as soon as the blocks it reads are final, and the factors do not depend on
`ntasks`; `ntasks=1` runs on the calling task. With `ntasks > 1`, use a single
BLAS thread (`BLAS.set_num_threads(1)`). `batchrank` bounds the pending update
columns per far block. One workspace serves one factorization at a time;
`LUWorkspace(workspace)` plans a second one for concurrent factorizations.
"""
mutable struct LUWorkspace{T,W,L,U,S,R}
    view::W
    leaves::Vector{DenseLeafLU{T,Matrix{T}}}
    lower::L
    upper::U
    steps::Vector{S}
    tasks::Vector{_LUTask}
    graph::_TaskGraph
    remaining::Vector{Threads.Atomic{Int}}
    treeswaps::Vector{Int}
    workers::Vector{_Worker{T,R}}
    batchrank::Int
end

# Blocks read by owner `j` of a solve plan: its diagonal subfactor and, for an
# update, the coefficient and earlier solved owners.
function _solvereads(P, j)
    H = P.subfactors[P.submap[j]]
    reads = copy(H.entries)
    append!(reads, (H.leafrecords[i] for i in H.leafspan))
    u = P.updatemap[j]
    if !iszero(u)
        append!(reads, P.products[u].a.entries)
        append!(reads, P.products[u].b.entries)
    end
    return unique!(sort!(reads))
end

function _panelreads!(reads, panels)
    for panel in panels
        push!(reads, panel.entry)
        append!(reads, panel.entries)
    end
    return reads
end

# Tasks and dependency graph of the factorization. Resources are the physical
# blocks of the view; Schur nodes follow their parent node and destinations
# their nearest node.
function _lugraph(W, lower, steps, instructions)
    builder = _TaskGraphBuilder(length(W.csr.storageidx))
    tasks = _LUTask[]
    function add!(task, reads, writes, after=())
        push!(tasks, task)
        return _addtask!(builder, reads, writes, after)
    end
    for instruction in instructions
        if instruction > 0
            leaf = lower.leafrecords[instruction]
            add!(_LUTask(_LULEAF, instruction, 0), (leaf,), (leaf,))
            continue
        end
        s = -instruction
        step = steps[s]
        L = step.pivots.factor
        leaves = [L.leafrecords[i] for i in L.leafspan]
        writes = collect(Int, step.pivots.rhs.entries)
        step.update && append!(writes, step.upper.rhs.entries)
        add!(_LUTask(_LUPIVOTS, s, 0), leaves, unique!(sort!(writes)))
        step.update || continue
        for (kind, P) in ((_LUUPPER, step.upper), (_LULOWER, step.lower))
            for j in eachindex(P.entries)
                add!(_LUTask(kind, s, j), _solvereads(P, j), (P.entries[j],))
            end
        end
        P = step.schur
        nodetasks = zeros(Int, length(P.nodes))
        for (n, node) in enumerate(P.nodes)
            reads = _panelreads!(_panelreads!(Int[], node.afar), node.bfar)
            after = node.parent == 0 ? () : (nodetasks[node.parent],)
            nodetasks[n] = add!(_LUTask(_LUNODE, s, n), unique!(sort!(reads)), (), after)
        end
        for (i, g) in enumerate(P.groups)
            reads = _panelreads!(_panelreads!(vcat(g.aentries, g.bentries), g.afar), g.bfar)
            after = g.node == 0 ? () : (nodetasks[g.node],)
            add!(_LUTask(_LUGROUP, s, i), unique!(sort!(reads)), (g.destination,), after)
        end
    end
    return tasks, _TaskGraph(builder)
end

function _luworkspace(
    W::_TreeView{T}; batchrank::Integer=128, ntasks::Integer=Threads.nthreads()
) where {T<:LinearAlgebra.BlasFloat}
    size(W, 1) == size(W, 2) || throw(DimensionMismatch("LU requires a square matrix"))
    ntasks = _checkntasks(ntasks)
    leaves = _luleaves(W)
    lower = HMatrixTriangular(W, leaves; uplo=:lower, check=false)
    # The triangular constructor canonicalizes leaf order by tree position.
    # Keep that exact order for the signed instruction stream.
    leaves = lower.leaves
    upper = HMatrixTriangular(W, leaves; uplo=:upper, check=false)
    rawsteps, instructions = Any[], Int[]
    tree = testtree(W.tree)

    function build!(node)
        leaf = lower.leafmap[node]
        if !iszero(leaf)
            push!(instructions, leaf)
            return nothing
        end
        children = collect(Int, _clusterchildren(tree, node))
        sort!(children; by=child -> first(W.rowranges[child]))
        isempty(children) &&
            throw(ArgumentError("missing LU diagonal leaf at cluster $node"))
        noderange = W.rowranges[node]
        for child in children
            childrange = W.rowranges[child]
            build!(child)
            # Unary geometric nodes can have the same occupied range. Their
            # aggregate permutation must not be applied a second time.
            childrange == noderange && continue
            prefix = first(noderange):(first(childrange) - 1)
            later = (last(childrange) + 1):last(noderange)
            L = HMatrixTriangular(lower; node=child)
            U = HMatrixTriangular(upper; node=child)
            prefixplan = _LUPivotPlan(L, blockview(W, childrange, prefix))
            upperblock = blockview(W, childrange, later)
            lowerblock = blockview(W, later, childrange)
            trailing = blockview(W, later, later)
            schur = _ProductPlan(trailing, lowerblock, upperblock; batchrank)
            upperplan = _SolvePlan(L, upperblock; side=:left, batchrank)
            lowerplan = _SolvePlan(U, lowerblock; side=:right, batchrank)
            step = _LUEliminationStep(
                prefixplan, upperplan, lowerplan, schur, !isempty(later)
            )
            push!(rawsteps, step)
            push!(instructions, -length(rawsteps))
        end
        return nothing
    end

    build!(_clusterroot(tree))
    steps =
        isempty(rawsteps) ? Nothing[] : convert(Vector{typeof(first(rawsteps))}, rawsteps)
    tasks, graph = _lugraph(W, lower, steps, instructions)
    swaps = _treeswaps(W.rowperm, size(W, 1))
    workers = [_Worker{T}() for _ in 1:ntasks]
    return LUWorkspace(
        W,
        leaves,
        lower,
        upper,
        steps,
        tasks,
        graph,
        [Threads.Atomic{Int}(0) for _ in steps],
        swaps,
        workers,
        Int(batchrank),
    )
end

LUWorkspace(H::HMatrixCore; kwargs...) = _luworkspace(_TreeView(H); kwargs...)
LUWorkspace(ws::LUWorkspace) =
    _luworkspace(_copyview(ws.view); batchrank=ws.batchrank, ntasks=length(ws.workers))

"""
    HMatrixLU <: Factorization

Hierarchical LU factorization `P*A ≈ L*U` returned by `lu` and `lu!`. `P` is the
product of the row permutations inside the dense diagonal leaves. The factors
live in the blocks of the factorized matrix. Solve with `F \\ b`, `ldiv!(F, b)`
or `ldiv!(x, F, b)`, also with `transpose(F)` and `F'`, for vectors and matrices
of right-hand sides. `F` is read-only during solves: concurrent solves with one
factorization each use their own [`SolveWorkspace`](@ref).

`converged`, `rank_limited`, `max_block_error` and `flushes` summarize the local
recompressions of the factorization, not a global error bound.
"""
struct HMatrixLU{T,W,L,U,R} <: LinearAlgebra.Factorization{T}
    view::W
    leaves::Vector{DenseLeafLU{T,Matrix{T}}}
    lower::L
    upper::U
    treeswaps::Vector{Int}
    converged::Bool
    rank_limited::Bool
    max_block_error::R
    flushes::Int
end

Base.size(F::HMatrixLU) = size(F.view)
Base.size(F::HMatrixLU, dim::Integer) = size(F.view, dim)
LinearAlgebra.issuccess(F::HMatrixLU) = all(issuccess, F.leaves)

# Bind the plan's view and leaves to `H`, with fresh pivot storage that the
# returned factorization takes over.
function _bindlu!(ws::LUWorkspace, H::HMatrixCore)
    _bind!(ws.view, H)
    W = ws.view
    for (i, leaf) in enumerate(ws.leaves)
        leaf.factors = W.near[W.csr.storageidx[ws.lower.leafrecords[i]]]
        leaf.ipiv = Vector{BlasInt}(undef, size(leaf.factors, 1))
        leaf.info = BlasInt(-1)
    end
    return ws
end

# One block operation of the factorization. Returns false at a singular leaf.
function _runlutask!(ws::LUWorkspace{T}, t, worker, options, check, checkfinite) where {T}
    task = ws.tasks[t]
    if task.kind == _LULEAF
        leaf = ws.leaves[task.step]
        leaflu!(leaf; check, checkfinite)
        return issuccess(leaf)
    end
    step = ws.steps[task.step]
    scratch, status = worker.scratch, worker.status
    if task.kind == _LUPIVOTS
        _executelupivots!(step.pivots)
        step.update && _applypivots!(step.upper, false)
    elseif task.kind == _LUUPPER
        _executesolveoperation!(step.upper, task.item, scratch, status, options)
    elseif task.kind == _LULOWER
        _executesolveoperation!(step.lower, task.item, scratch, status, options)
    elseif task.kind == _LUNODE
        _executenode!(step.schur, task.item, -one(T), scratch, status, options)
    else
        P = step.schur
        _executegroup!(P, P.groups[task.item], -one(T), scratch, status, options)
        # The last destination releases the transient node factors.
        isone(Threads.atomic_sub!(ws.remaining[task.step], 1)) && empty!(P.results)
    end
    return true
end

# Execute the recursive block LU on the bound matrix. Each subtree factors its
# leading diagonal block, applies its leaf pivots once to the completed lower
# and pending upper panels, performs pivot-free hierarchical triangular solves
# and accumulates A22 -= L21*U12. The block operations run as their inputs
# become final. Stops at the first singular leaf when `check=false`.
function _factorize!(ws::LUWorkspace{T}, options, check, checkfinite) where {T}
    W = ws.view
    for k in eachindex(W.csr.storageidx)
        _checkworkspaceblock(W, k)
    end
    _checkworkspacealias(W, W, first(ws.workers).scratch)
    for (s, step) in enumerate(ws.steps)
        resize!(step.schur.results, length(step.schur.nodes))
        ws.remaining[s][] = length(step.schur.groups)
    end
    _resetstatus!(ws.workers)
    try
        _rungraph!(ws.graph, ws.workers) do t, worker
            return _runlutask!(ws, t, worker, options, check, checkfinite)
        end
    finally
        foreach(step -> empty!(step.schur.results), ws.steps)
    end
    return _workerstatus(ws.workers)
end

function _rebind(H::HMatrixTriangular, W, leaves)
    panels = map(H.panels) do p
        return _TriangularPanel(
            p.row, p.col, HMatrixBlock(W, p.block.rows, p.block.cols, p.block.entries)
        )
    end
    return HMatrixTriangular(
        W,
        leaves,
        H.leafrecords,
        H.leafmap,
        panels,
        H.leftptr,
        H.leftmap,
        H.rightptr,
        H.rightmap,
        H.entries,
        H.uplo,
        H.node,
        H.range,
        H.leafspan,
    )
end

"""
    lu!(H::HMatrixCore[, workspace::LUWorkspace]; check=true)

Hierarchical LU factorization of `H` in place: the blocks of `H` are
overwritten by the factors, so `H` no longer represents the original matrix.
Every block update is recompressed to `H.tol`. A [`LUWorkspace`](@ref) from an
earlier factorization of a matrix with the same partition avoids planning
again. With `check=false`, a singular dense diagonal leaf returns a factorization
with `issuccess(F) == false` instead of throwing a [`LeafLUFailure`](@ref).
Requires full (not symmetric half-) storage.
"""
function LinearAlgebra.lu!(
    H::HMatrixCore{T}, ws::LUWorkspace{T}; check::Bool=true, checkfinite::Bool=true
) where {T<:LinearAlgebra.BlasFloat}
    _bindlu!(ws, H)
    options = (; atol=0, rtol=H.tol, maxrank=typemax(Int))
    status = _factorize!(ws, options, check, checkfinite)
    # The factorization owns its own view and leaf objects, so the workspace can
    # move on to another matrix without invalidating it.
    W = _copyview(ws.view)
    leaves = [DenseLeafLU(l.factors, l.ipiv, l.info, l.node) for l in ws.leaves]
    return HMatrixLU(
        W,
        leaves,
        _rebind(ws.lower, W, leaves),
        _rebind(ws.upper, W, leaves),
        ws.treeswaps,
        status.converged,
        status.rank_limited,
        status.max_block_error,
        status.flushes,
    )
end

function LinearAlgebra.lu!(H::HMatrixCore; kwargs...)
    @warn "lu! overwrites the HMatrix with its LU factors; use lu to keep it." maxlog = 1
    return lu!(H, LUWorkspace(H); kwargs...)
end

"""
    lu(H::HMatrixCore[, workspace::LUWorkspace]; check=true)

Hierarchical LU factorization of a copy of `H`; `H` is unchanged. See `lu!`.
"""
LinearAlgebra.lu(H::HMatrixCore, ws::LUWorkspace; kwargs...) = lu!(copy(H), ws; kwargs...)
function LinearAlgebra.lu(H::HMatrixCore; kwargs...)
    A = copy(H)
    return lu!(A, LUWorkspace(A); kwargs...)
end

"""
    SolveWorkspace(F::HMatrixLU)
    SolveWorkspace(workspace::SolveWorkspace)

Buffers for solves with the factorization `F`. It acts like `F`:
`ldiv!(S, b)`, `ldiv!(x, S, b)`, `S \\ b`, `ldiv!(transpose(S), b)` and
`ldiv!(S', b)`, for vectors and matrices of right-hand sides. Solves with `F`
itself build a temporary workspace on every call. One workspace serves one
solve at a time; `SolveWorkspace(workspace)` gives another task its own.
"""
struct SolveWorkspace{T,F<:HMatrixLU{T}} <: LinearAlgebra.Factorization{T}
    factor::F
    scratch::BlockScratch{T}
end

SolveWorkspace(F::HMatrixLU{T}) where {T} = SolveWorkspace(F, BlockScratch{T}())
SolveWorkspace(S::SolveWorkspace) = SolveWorkspace(S.factor)

Base.size(S::SolveWorkspace) = size(S.factor)
Base.size(S::SolveWorkspace, dim::Integer) = size(S.factor, dim)
LinearAlgebra.issuccess(S::SolveWorkspace) = issuccess(S.factor)

# Solve op(A)*X = B in place for B in external order. A = P'*L*U in tree order,
# so A^T = U^T*L^T*P.
function _lusolve!(F::HMatrixLU{T}, B::StridedMatrix{T}, scratch, transposed) where {T}
    issuccess(F) || throw(ArgumentError("hierarchical LU factorization failed"))
    size(B, 1) == size(F, 1) || throw(DimensionMismatch("right-hand side size differs"))
    _applytreeswaps!(B, F.treeswaps, false)
    try
        if transposed
            _trisolve!(B, F.upper, scratch, Val(:T), Val(:left))
            _trisolve!(B, F.lower, scratch, Val(:T), Val(:left))
            _pivot!(B, F.lower; inverse=true)
        else
            _pivot!(B, F.lower)
            _trisolve!(B, F.lower, scratch, Val(:N), Val(:left))
            _trisolve!(B, F.upper, scratch, Val(:N), Val(:left))
        end
    finally
        _applytreeswaps!(B, F.treeswaps, true)
    end
    return B
end

_solvematrix(B::StridedMatrix) = B
_solvematrix(B::StridedVector) = reshape(B, :, 1)

_solveparts(S::SolveWorkspace) = (S.factor, S.scratch)
function _solveparts(F::HMatrixLU{T}) where {T}
    @warn "Solving with an HMatrixLU builds a temporary SolveWorkspace on every " *
        "call. Construct `AdaptiveCrossApproximation.SolveWorkspace(F)` once and " *
        "solve with it instead." maxlog = 1
    return (F, BlockScratch{T}())
end

const _LUSolver{T} = Union{HMatrixLU{T},SolveWorkspace{T}}

function LinearAlgebra.ldiv!(S::_LUSolver{T}, B::StridedVecOrMat{T}) where {T}
    F, scratch = _solveparts(S)
    _lusolve!(F, _solvematrix(B), scratch, false)
    return B
end

function LinearAlgebra.ldiv!(
    S::LinearAlgebra.TransposeFactorization{T,<:_LUSolver{T}}, B::StridedVecOrMat{T}
) where {T}
    F, scratch = _solveparts(Base.parent(S))
    _lusolve!(F, _solvematrix(B), scratch, true)
    return B
end

# A^H x = b is solved as A^T conj(x) = conj(b); for real matrices A^H = A^T.
function LinearAlgebra.ldiv!(
    S::LinearAlgebra.AdjointFactorization{T,<:_LUSolver{T}}, B::StridedVecOrMat{T}
) where {T}
    F, scratch = _solveparts(Base.parent(S))
    T <: Real || conj!(B)
    _lusolve!(F, _solvematrix(B), scratch, true)
    T <: Real || conj!(B)
    return B
end
