# Output partition of one product direction. Chunks are disjoint, contiguous
# tree-order ranges of balanced cost; `entries[ptr[c]:ptr[c+1]-1]` are the CSR
# entries whose output range overlaps chunk `c`. Blocks crossing a chunk border
# are split by output rows, so chunks never write the same entries.
struct _MatVecPlan
    ranges::Vector{UnitRange{Int}}
    ptr::Vector{Int}
    entries::Vector{Int}
end

mutable struct _MatVecBuffers{T}
    x::Vector{T}
    y::Vector{T}
    z::Vector{T}
    offsets::Vector{Int}
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
    buffers::_MatVecBuffers{T}
end

Base.size(A::MatVecWorkspace) = size(A.matrix)
Base.size(A::MatVecWorkspace, dim::Integer) = size(A.matrix, dim)

function _blockcost(core, k, rows, cols)
    if core.csr.isfar[k]
        return Float64(rank(core.far[core.csr.storageidx[k]])) *
               (length(rows) + length(cols))
    end
    return Float64(length(rows)) * length(cols)
end

# Spread every block's cost evenly over its output range, then cut the tree-order
# axis into chunks of equal cumulative cost.
function _matvecplan(core, index, entries, output::Val{O}, nchunks) where {O}
    n = O === :rows ? index.dim[1] : index.dim[2]
    density = zeros(Float64, n + 1)
    for k in entries
        rows, cols = _rowrange(index, k), _colrange(index, k)
        out = O === :rows ? rows : cols
        isempty(out) && continue
        # A small per-block term accounts for kernel-call overhead.
        w = (_blockcost(core, k, rows, cols) + 64) / length(out)
        density[first(out)] += w
        density[last(out) + 1] -= w
    end
    cumulative = Vector{Float64}(undef, n)
    rate, total = 0.0, 0.0
    for i in 1:n
        rate += density[i]
        total += rate
        cumulative[i] = total
    end
    ranges = UnitRange{Int}[]
    start = 1
    for c in 1:nchunks
        start > n && break
        stop = if c == nchunks
            n
        else
            max(start, searchsortedfirst(cumulative, total * c / nchunks))
        end
        stop = min(stop, n)
        push!(ranges, start:stop)
        start = stop + 1
    end
    start <= n && push!(ranges, start:n)
    starts = first.(ranges)
    lists = [Int[] for _ in ranges]
    for k in entries
        out = O === :rows ? _rowrange(index, k) : _colrange(index, k)
        isempty(out) && continue
        c = searchsortedlast(starts, first(out))
        while c <= length(ranges) && first(ranges[c]) <= last(out)
            push!(lists[c], k)
            c += 1
        end
    end
    ptr = [1]
    for list in lists
        push!(ptr, ptr[end] + length(list))
    end
    return _MatVecPlan(ranges, ptr, reduce(vcat, lists; init=Int[]))
end

# Contiguous groups of far entries with equal factor-application cost.
function _farchunks(core, index, farentries, nchunks)
    costs = map(farentries) do k
        return rank(core.far[core.csr.storageidx[k]]) *
               (length(_rowrange(index, k)) + length(_colrange(index, k))) + 64.0
    end
    return _costchunks(costs, nchunks)
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
    rows = _matvecplan(H, index, entries, Val(:rows), nchunks)
    cols = _matvecplan(H, index, entries, Val(:cols), nchunks)
    farptr = _farchunks(H, index, farentries, nchunks)
    buffers = _MatVecBuffers{T}(T[], T[], T[], zeros(Int, length(csr)))
    return MatVecWorkspace(H, index, rows, cols, farentries, farptr, scheduler, buffers)
end

MatVecWorkspace(C::HMatrixComponent{<:Any,<:Any,Far}; kwargs...) where {Far} =
    MatVecWorkspace(C.parent, Far; kwargs...)

function MatVecWorkspace(A::MatVecWorkspace{T}) where {T}
    buffers = _MatVecBuffers{T}(T[], T[], T[], zeros(Int, length(A.buffers.offsets)))
    return MatVecWorkspace(
        A.matrix, A.index, A.rows, A.cols, A.farentries, A.farptr, A.scheduler, buffers
    )
end

# Column access for vectors and matrices alike, without reshaping wrappers.
@inline _vecget(x::AbstractVector, i, c) = @inbounds x[i]
@inline _vecget(x::AbstractMatrix, i, c) = @inbounds x[i, c]
@inline _vecset!(y::AbstractVector, v, i, c) = @inbounds y[i] = v
@inline _vecset!(y::AbstractMatrix, v, i, c) = @inbounds y[i, c] = v

# A real matrix acting on complex vectors works on real and imaginary parts as
# separate tree-order columns.
_splitcomplex(::Type{T}, ::Type{S}) where {T,S} = T <: Real && S <: Complex

function _gather!(xt, x, perm, range, k, conjugate, ::Val{true})
    for c in 1:k, i in range
        v = _vecget(x, _permindex(perm, i), c)
        @inbounds xt[i, 2c - 1] = real(v)
        @inbounds xt[i, 2c] = conjugate ? -imag(v) : imag(v)
    end
    return nothing
end
function _gather!(xt, x, perm, range, k, conjugate, ::Val{false})
    for c in 1:k, i in range
        v = _vecget(x, _permindex(perm, i), c)
        @inbounds xt[i, c] = conjugate ? conj(v) : v
    end
    return nothing
end

@inline _treevalue(yt, i, c, ::Val{true}) = @inbounds complex(yt[i, 2c - 1], yt[i, 2c])
@inline _treevalue(yt, i, c, ::Val{false}) = @inbounds yt[i, c]

function _scatter!(y, yt, perm, range, k, alpha, beta, conjugate, split)
    for c in 1:k, i in range
        v = _treevalue(yt, i, c, split)
        conjugate && (v = conj(v))
        p = _permindex(perm, i)
        if iszero(beta)
            _vecset!(y, alpha * v, p, c)
        else
            _vecset!(y, alpha * v + beta * _vecget(y, p, c), p, c)
        end
    end
    return nothing
end

# Phase 1: z = V*x (or U^T*x for flipped blocks) for every far entry of a chunk.
function _matvecfar!(A, xt, z, chunk, ::Val{O}) where {O}
    core, index = A.matrix, A.index
    csr, offsets = core.csr, A.buffers.offsets
    width = size(xt, 2)
    for j in A.farptr[chunk]:(A.farptr[chunk + 1] - 1)
        k = A.farentries[j]
        R = core.far[csr.storageidx[k]]
        r = rank(R)
        r == 0 && continue
        input = O === :N ? _colrange(index, k) : _rowrange(index, k)
        Z = reshape(view(z, (offsets[k] + 1):(offsets[k] + r * width)), r, width)
        X = view(xt, input, :)
        if xor(O === :T, csr.transposed[k])
            mul!(Z, transpose(R.U), X)
        else
            mul!(Z, R.V, X)
        end
    end
    return nothing
end

@inline function _matvecdense!(Y, S, X, local_, flip)
    if flip
        mul!(Y, transpose(view(S, :, local_)), X, true, true)
    else
        mul!(Y, view(S, local_, :), X, true, true)
    end
    return nothing
end

@inline function _matveclowrank!(Y, R, Z, local_, flip)
    if flip
        mul!(Y, transpose(view(R.V, :, local_)), Z, true, true)
    else
        mul!(Y, view(R.U, local_, :), Z, true, true)
    end
    return nothing
end

# Phase 2: accumulate one output chunk in tree order.
function _matvecchunk!(A, yt, xt, z, plan, chunk, ::Val{O}) where {O}
    core, index = A.matrix, A.index
    csr, offsets = core.csr, A.buffers.offsets
    width = size(xt, 2)
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
            Z = reshape(view(z, (offsets[k] + 1):(offsets[k] + r * width)), r, width)
            if width == 1
                _matveclowrank!(view(yt, overlap, 1), R, view(Z, :, 1), local_, flip)
            else
                _matveclowrank!(view(yt, overlap, :), R, Z, local_, flip)
            end
        else
            S = core.near[csr.storageidx[k]]
            if width == 1
                _matvecdense!(view(yt, overlap, 1), S, view(xt, input, 1), local_, flip)
            else
                _matvecdense!(view(yt, overlap, :), S, view(xt, input, :), local_, flip)
            end
        end
    end
    return nothing
end

function _reservematvec!(A::MatVecWorkspace{T}, n, m, width) where {T}
    buffers = A.buffers
    core, csr = A.matrix, A.matrix.csr
    total = 0
    for k in A.farentries
        buffers.offsets[k] = total
        total += rank(core.far[csr.storageidx[k]]) * width
    end
    length(buffers.x) < n * width && resize!(buffers.x, n * width)
    length(buffers.y) < m * width && resize!(buffers.y, m * width)
    length(buffers.z) < total && resize!(buffers.z, total)
    return nothing
end

function _matvec!(
    y, A::MatVecWorkspace{T}, x, alpha, beta, op::Val{O}, conjugate
) where {T,O}
    m, n = O === :N ? size(A) : reverse(size(A))
    size(y, 1) == m && size(x, 1) == n && size(y, 2) == size(x, 2) ||
        throw(DimensionMismatch("incompatible matrix-vector product dimensions"))
    split = _splitcomplex(T, eltype(x)) || _splitcomplex(T, eltype(y))
    return _matvec!(y, A, x, alpha, beta, op, conjugate, Val(split))
end

function _matvec!(
    y, A::MatVecWorkspace{T}, x, alpha, beta, op::Val{O}, conjugate, split::Val{S}
) where {T,O,S}
    m, n = O === :N ? size(A) : reverse(size(A))
    k = size(x, 2)
    width = S ? 2k : k
    if iszero(alpha) || iszero(width) || iszero(n)
        iszero(beta) ? fill!(y, zero(eltype(y))) : (isone(beta) || rmul!(y, beta))
        return y
    end
    _reservematvec!(A, n, m, width)
    xt = reshape(view(A.buffers.x, 1:(n * width)), n, width)
    yt = reshape(view(A.buffers.y, 1:(m * width)), m, width)
    z = A.buffers.z
    index = A.index
    inperm, outperm =
        O === :N ? (index.colperm, index.rowperm) : (index.rowperm, index.colperm)
    inplan = O === :N ? A.cols : A.rows
    plan = O === :N ? A.rows : A.cols
    scheduler = A.scheduler
    tforeach(eachindex(inplan.ranges); scheduler) do c
        return _gather!(xt, x, inperm, inplan.ranges[c], k, conjugate, split)
    end
    tforeach(1:(length(A.farptr) - 1); scheduler) do c
        return _matvecfar!(A, xt, z, c, op)
    end
    tforeach(eachindex(plan.ranges); scheduler) do c
        _matvecchunk!(A, yt, xt, z, plan, c, op)
        return _scatter!(y, yt, outperm, plan.ranges[c], k, alpha, beta, conjugate, split)
    end
    return y
end

# Adjoint products use (H^T * conj(x))' without a separate conjugated code path.
_matvecop(A::MatVecWorkspace) = (A, Val(:N), false)
_matvecop(A::LinearMaps.TransposeMap{<:Any,<:MatVecWorkspace}) = (A.lmap, Val(:T), false)
_matvecop(A::LinearMaps.AdjointMap{<:Any,<:MatVecWorkspace}) = (A.lmap, Val(:T), true)

const _MatVecOperator = Union{
    MatVecWorkspace,
    LinearMaps.TransposeMap{<:Any,<:MatVecWorkspace},
    LinearMaps.AdjointMap{<:Any,<:MatVecWorkspace},
}

function _workspacemul!(y, A, x, alpha, beta)
    W, op, conjugate = _matvecop(A)
    return _matvec!(y, W, x, alpha, beta, op, conjugate)
end

# Products with the matrix itself build a temporary workspace on every call.
const _HMatrixOperator = Union{
    HMatrixCore,
    HMatrixComponent,
    LinearMaps.TransposeMap{<:Any,<:Union{HMatrixCore,HMatrixComponent}},
    LinearMaps.AdjointMap{<:Any,<:Union{HMatrixCore,HMatrixComponent}},
}

_temporaryworkspace(A::Union{HMatrixCore,HMatrixComponent}) = MatVecWorkspace(A)
_temporaryworkspace(A::LinearMaps.TransposeMap) = transpose(MatVecWorkspace(A.lmap))
_temporaryworkspace(A::LinearMaps.AdjointMap) = adjoint(MatVecWorkspace(A.lmap))

function _temporarymul!(y, A, x, alpha, beta)
    @warn "Multiplying with an HMatrix builds a temporary MatVecWorkspace on every " *
        "call. Construct `AdaptiveCrossApproximation.MatVecWorkspace(H)` once and " *
        "multiply with it instead." maxlog = 1
    return _workspacemul!(y, _temporaryworkspace(A), x, alpha, beta)
end

# LinearMaps dispatches vectors and matrices separately; matching both avoids
# method ambiguities with its generic wrapper methods.
for In in (AbstractVector, AbstractMatrix)
    @eval begin
        LinearMaps._unsafe_mul!(y, A::_MatVecOperator, x::$In, alpha, beta) =
            _workspacemul!(y, A, x, alpha, beta)
        LinearMaps._unsafe_mul!(y, A::_MatVecOperator, x::$In) =
            _workspacemul!(y, A, x, true, false)
        LinearMaps._unsafe_mul!(y, A::_HMatrixOperator, x::$In, alpha, beta) =
            _temporarymul!(y, A, x, alpha, beta)
        LinearMaps._unsafe_mul!(y, A::_HMatrixOperator, x::$In) =
            _temporarymul!(y, A, x, true, false)
    end
end

# Several right-hand sides give a dense result, as for a matrix.
function Base.:*(A::Union{_MatVecOperator,_HMatrixOperator}, X::AbstractMatrix)
    Y = similar(X, promote_type(eltype(A), eltype(X)), size(A, 1), size(X, 2))
    return mul!(Y, A, X)
end
