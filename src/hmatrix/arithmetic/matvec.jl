"""
    _MatVecPlan

Output partition of one product direction of a [`MatVecWorkspace`](@ref).
Chunks are disjoint, contiguous tree-order ranges of balanced cost;
`entries[ptr[c]:(ptr[c + 1] - 1)]` are the CSR entries whose output range
overlaps chunk `c`. Blocks crossing a chunk border are split by output rows, so
parallel chunks never write the same entries and need no locking.
"""
struct _MatVecPlan
    ranges::Vector{UnitRange{Int}}
    ptr::Vector{Int}
    entries::Vector{Int}
end

"""
    MatVecWorkspace(H::HMatrixCore; ntasks=Threads.nthreads())
    MatVecWorkspace(workspace::MatVecWorkspace)

Matrix-vector product operator for `H`. It holds `H`, a precomputed parallel
schedule and the internal buffers of the product, and can be used wherever `H`
is used as a linear map:

```julia
A = AdaptiveCrossApproximation.MatVecWorkspace(H)
mul!(y, A, x)                 # y = H*x
mul!(y, transpose(A), x, α, β)
Y = A' * X                    # several right-hand sides at once
```

Vectors are in the original (external) ordering. The workspace owns its
buffers: repeated products with the same number of columns do not allocate
numerical storage, and the work of one product is distributed over `ntasks`
parallel tasks (`ntasks=1` runs on the calling task). One workspace
serves one product at a time. For concurrent products, give each task its own
copy with `MatVecWorkspace(workspace)`; copies share the schedule and `H`.
Products with `H` itself build a temporary workspace on every call.

A real matrix also multiplies complex vectors. The adjoint product of a complex
matrix uses the conjugated transpose product.
"""
struct MatVecWorkspace{T,M<:HMatrixCore{T},I,S} <: LinearMaps.LinearMap{T}
    matrix::M
    index::I
    rows::_MatVecPlan
    cols::_MatVecPlan
    farentries::Vector{Int}
    farptr::Vector{Int}
    scheduler::S
    # Input and output in tree order, and `V*x` of every far block at `offsets`.
    x::Vector{T}
    y::Vector{T}
    z::Vector{T}
    offsets::Vector{Int}
end

function MatVecWorkspace(
    H::HMatrixCore{T},
    farfilter::Union{Nothing,Bool}=nothing;
    ntasks::Integer=Threads.nthreads(),
) where {T}
    scheduler, nchunks = _scheduler(ntasks), _nchunks(ntasks)
    index = _TreeIndex(H)
    csr = H.csr
    entries = filter(k -> isnothing(farfilter) || csr.isfar[k] == farfilter, 1:length(csr))
    farentries = filter(k -> csr.isfar[k], entries)
    rows = _MatVecPlan(H, index, entries, Val(:rows), nchunks)
    cols = _MatVecPlan(H, index, entries, Val(:cols), nchunks)
    # Contiguous groups of far entries with equal factor-application cost.
    farptr = _costchunks([_matveccost(H, index, k) + 64 for k in farentries], nchunks)
    return MatVecWorkspace(
        H,
        index,
        rows,
        cols,
        farentries,
        farptr,
        scheduler,
        T[],
        T[],
        T[],
        zeros(Int, length(csr)),
    )
end

MatVecWorkspace(C::HMatrixComponent{<:Any,<:Any,Far}; kwargs...) where {Far} =
    MatVecWorkspace(C.parent, Far; kwargs...)

function MatVecWorkspace(A::MatVecWorkspace{T}) where {T}
    return MatVecWorkspace(
        A.matrix,
        A.index,
        A.rows,
        A.cols,
        A.farentries,
        A.farptr,
        A.scheduler,
        T[],
        T[],
        T[],
        zeros(Int, length(A.offsets)),
    )
end

Base.size(A::MatVecWorkspace) = size(A.matrix)
Base.size(A::MatVecWorkspace, dim::Integer) = size(A.matrix, dim)

# Floating-point operations of one product with block `k`.
function _matveccost(H, index, k)
    rows, cols = length(_rowrange(index, k)), length(_colrange(index, k))
    H.csr.isfar[k] || return Float64(rows) * cols
    return Float64(rank(H.far[H.csr.storageidx[k]])) * (rows + cols)
end

# Spread every block's cost evenly over its output range, then cut the tree-order
# axis into chunks of equal cost.
function _MatVecPlan(H, index, entries, ::Val{O}, nchunks) where {O}
    output(k) = O === :rows ? _rowrange(index, k) : _colrange(index, k)
    n = O === :rows ? index.dim[1] : index.dim[2]
    density = zeros(Float64, n + 1)
    for k in entries
        out = output(k)
        isempty(out) && continue
        # A small per-block term accounts for kernel-call overhead.
        w = (_matveccost(H, index, k) + 64) / length(out)
        density[first(out)] += w
        density[last(out) + 1] -= w
    end
    chunkptr = _costchunks(cumsum(view(density, 1:n)), nchunks)
    lists = [Int[] for _ in 1:(length(chunkptr) - 1)]
    for k in entries
        out = output(k)
        isempty(out) && continue
        for c in
            searchsortedlast(chunkptr, first(out)):searchsortedlast(chunkptr, last(out))

            push!(lists[c], k)
        end
    end
    ranges = [chunkptr[c]:(chunkptr[c + 1] - 1) for c in eachindex(lists)]
    return _MatVecPlan(ranges, linearizestorage(lists)...)
end

function _matvec!(y, A::MatVecWorkspace{T}, x, alpha, beta, ::Val{O}, conjugate) where {T,O}
    m, n = O === :N ? size(A) : reverse(size(A))
    size(y, 1) == m && size(x, 1) == n && size(y, 2) == size(x, 2) ||
        throw(DimensionMismatch("incompatible matrix-vector product dimensions"))
    # A real matrix acting on complex vectors works on their real and imaginary
    # parts as separate columns.
    S = T <: Real && (eltype(x) <: Complex || eltype(y) <: Complex)
    split = Val(S)
    width = S ? 2size(x, 2) : size(x, 2)
    if iszero(alpha) || iszero(width) || iszero(n)
        iszero(beta) ? fill!(y, zero(eltype(y))) : (isone(beta) || rmul!(y, beta))
        return y
    end
    X = x isa AbstractVector ? reshape(x, :, 1) : x
    Y = y isa AbstractVector ? reshape(y, :, 1) : y
    _reservematvec!(A, n, m, width)
    xt = reshape(view(A.x, 1:(n * width)), n, width)
    yt = reshape(view(A.y, 1:(m * width)), m, width)
    index = A.index
    inperm, outperm =
        O === :N ? (index.colperm, index.rowperm) : (index.rowperm, index.colperm)
    inplan, outplan = O === :N ? (A.cols, A.rows) : (A.rows, A.cols)
    tforeach(eachindex(inplan.ranges); A.scheduler) do c
        return _gather!(xt, X, inperm, inplan.ranges[c], conjugate, split)
    end
    tforeach(1:(length(A.farptr) - 1); A.scheduler) do c
        return _matvecfar!(A, xt, c, Val(O))
    end
    tforeach(eachindex(outplan.ranges); A.scheduler) do c
        _matvecchunk!(A, yt, xt, outplan, c, Val(O))
        return _scatter!(Y, yt, outperm, outplan.ranges[c], alpha, beta, conjugate, split)
    end
    return y
end

function _reservematvec!(A::MatVecWorkspace, n, m, width)
    total = 0
    for k in A.farentries
        A.offsets[k] = total
        total += rank(A.matrix.far[A.matrix.csr.storageidx[k]]) * width
    end
    length(A.x) < n * width && resize!(A.x, n * width)
    length(A.y) < m * width && resize!(A.y, m * width)
    length(A.z) < total && resize!(A.z, total)
    return nothing
end

function _gather!(xt, x, perm, range, conjugate, ::Val{S}) where {S}
    @inbounds for c in axes(x, 2), i in range
        v = x[_permindex(perm, i), c]
        conjugate && (v = conj(v))
        if S
            xt[i, 2c - 1] = real(v)
            xt[i, 2c] = imag(v)
        else
            xt[i, c] = v
        end
    end
    return nothing
end

function _scatter!(y, yt, perm, range, alpha, beta, conjugate, ::Val{S}) where {S}
    @inbounds for c in axes(y, 2), i in range
        v = S ? complex(yt[i, 2c - 1], yt[i, 2c]) : yt[i, c]
        conjugate && (v = conj(v))
        p = _permindex(perm, i)
        y[p, c] = iszero(beta) ? alpha * v : alpha * v + beta * y[p, c]
    end
    return nothing
end

# The `V*x` buffer of far entry `k` for `width` columns.
function _farbuffer(A, k, r, width)
    return reshape(view(A.z, (A.offsets[k] + 1):(A.offsets[k] + r * width)), r, width)
end

# Phase 1: z = V*x (or U^T*x for flipped blocks) for every far entry of a chunk.
# Flipped products are formed as transpose(z) = transpose(x)*U: OpenBLAS serializes
# concurrent products with a transposed factor on a scratch-buffer lock.
function _matvecfar!(A, xt, chunk, ::Val{O}) where {O}
    core, index, csr = A.matrix, A.index, A.matrix.csr
    for j in A.farptr[chunk]:(A.farptr[chunk + 1] - 1)
        k = A.farentries[j]
        R = core.far[csr.storageidx[k]]
        r = rank(R)
        r == 0 && continue
        Z = _farbuffer(A, k, r, size(xt, 2))
        X = view(xt, O === :N ? _colrange(index, k) : _rowrange(index, k), :)
        if xor(O === :T, csr.transposed[k])
            mul!(transpose(Z), transpose(X), R.U)
        else
            mul!(Z, R.V, X)
        end
    end
    return nothing
end

# Phase 2: accumulate one output chunk in tree order. A block contributes the rows
# `local_` of `M` (dense block, or the output factor of a far block) times `X`.
function _matvecchunk!(A, yt, xt, plan, chunk, ::Val{O}) where {O}
    core, index, csr = A.matrix, A.index, A.matrix.csr
    range = plan.ranges[chunk]
    fill!(view(yt, range, :), zero(eltype(yt)))
    for j in plan.ptr[chunk]:(plan.ptr[chunk + 1] - 1)
        k = plan.entries[j]
        rows, cols = _rowrange(index, k), _colrange(index, k)
        output, input = O === :N ? (rows, cols) : (cols, rows)
        overlap = _overlap(output, range)
        local_ = _relative(overlap, output)
        flip = xor(O === :T, csr.transposed[k])
        if csr.isfar[k]
            R = core.far[csr.storageidx[k]]
            r = rank(R)
            r == 0 && continue
            M, X = flip ? R.V : R.U, _farbuffer(A, k, r, size(xt, 2))
            _addrows!(yt, overlap, M, X, local_, flip)
        else
            M, X = core.near[csr.storageidx[k]], view(xt, input, :)
            _addrows!(yt, overlap, M, X, local_, flip)
        end
    end
    return nothing
end

# yt[overlap, :] += op(M)[local_, :] * X, with BLAS gemv for a single column.
@inline function _addrows!(yt, overlap, M, X, local_, flip)
    if size(X, 2) == 1
        _addrows!(view(yt, overlap, 1), M, view(X, :, 1), local_, flip)
    else
        _addrows!(view(yt, overlap, :), M, X, local_, flip)
    end
    return nothing
end

# Flipped blocks with several columns as transpose(Y) += transpose(X)*M; see
# `_matvecfar!`.
@inline function _addrows!(Y, M, X, local_, flip)
    if !flip
        mul!(Y, view(M, local_, :), X, true, true)
    elseif Y isa AbstractVector
        mul!(Y, transpose(view(M, :, local_)), X, true, true)
    else
        mul!(transpose(Y), transpose(X), view(M, :, local_), true, true)
    end
    return nothing
end

# Workspace, direction and conjugation of every operator that can be multiplied.
# Adjoint products use (H^T * conj(x))' without a separate conjugated code path.
_matvecop(A::MatVecWorkspace) = (A, Val(:N), false)
_matvecop(A::LinearMaps.TransposeMap) = (first(_matvecop(A.lmap)), Val(:T), false)
_matvecop(A::LinearMaps.AdjointMap) = (first(_matvecop(A.lmap)), Val(:T), true)
function _matvecop(A::Union{HMatrixCore,HMatrixComponent})
    @warn "Multiplying with an HMatrix builds a temporary MatVecWorkspace on every " *
        "call. Construct `AdaptiveCrossApproximation.MatVecWorkspace(H)` once and " *
        "multiply with it instead." maxlog = 1
    return (MatVecWorkspace(A), Val(:N), false)
end

const _MatVecMap = Union{MatVecWorkspace,HMatrixCore,HMatrixComponent}
const _MatVecOperator = Union{
    _MatVecMap,
    LinearMaps.TransposeMap{<:Any,<:_MatVecMap},
    LinearMaps.AdjointMap{<:Any,<:_MatVecMap},
}

function _operatormul!(y, A, x, alpha, beta)
    W, op, conjugate = _matvecop(A)
    return _matvec!(y, W, x, alpha, beta, op, conjugate)
end

# LinearMaps dispatches vectors and matrices separately; matching both avoids
# method ambiguities with its generic wrapper methods.
for In in (AbstractVector, AbstractMatrix)
    @eval begin
        LinearMaps._unsafe_mul!(y, A::_MatVecOperator, x::$In, alpha, beta) =
            _operatormul!(y, A, x, alpha, beta)
        LinearMaps._unsafe_mul!(y, A::_MatVecOperator, x::$In) =
            _operatormul!(y, A, x, true, false)
    end
end

# Several right-hand sides give a dense result, as for a matrix.
function Base.:*(A::_MatVecOperator, X::AbstractMatrix)
    Y = similar(X, promote_type(eltype(A), eltype(X)), size(A, 1), size(X, 2))
    return mul!(Y, A, X)
end
