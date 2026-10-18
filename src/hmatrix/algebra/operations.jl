# Public hierarchical arithmetic on HMatrixCore. Every operation recompresses to
# the largest tolerance of the matrices involved.

_options(matrices...) =
    (; atol=0, rtol=maximum(H.tol for H in matrices), maxrank=typemax(Int))

function _withtol(H::HMatrixCore{T}, tol) where {T}
    return HMatrix{T}(H.tree, H.near, H.far, H.csr, H.csc, H.ordering, H.dim, tol)
end

function _checkeltypes(matrices...)
    T = eltype(first(matrices))
    all(H -> eltype(H) == T, matrices) ||
        throw(ArgumentError("H-matrix arithmetic requires equal element types"))
    return T
end

"""
    rmul!(H::HMatrixCore, α)
    lmul!(α, H::HMatrixCore)

Scale `H` in place by `α`. Exact; dense blocks are scaled and every low-rank
block scales its smaller factor. `α = 0` leaves rank-zero low-rank blocks.
"""
function LinearAlgebra.rmul!(H::HMatrixCore{T}, alpha::Number) where {T}
    alpha = convert(T, alpha)
    isone(alpha) && return H
    for D in H.near
        iszero(alpha) ? fill!(D, zero(T)) : rmul!(D, alpha)
    end
    foreach(R -> _scalelowrank!(R, alpha), H.far)
    return H
end
LinearAlgebra.lmul!(alpha::Number, H::HMatrixCore) = rmul!(H, alpha)

# Union{Real,Complex} rather than Number: LinearMaps defines lazy scaled maps
# for these scalars.
Base.:*(alpha::Union{Real,Complex}, H::HMatrixCore) = rmul!(copy(H), alpha)
Base.:*(H::HMatrixCore, alpha::Union{Real,Complex}) = rmul!(copy(H), alpha)
Base.:-(H::HMatrixCore) = rmul!(copy(H), -1)

"""
    AdditionWorkspace(C::HMatrixCore, A::HMatrixCore; ntasks=Threads.nthreads())

Buffers for `axpy!(α, A, C, workspace)` and `axpby!(α, A, β, C, workspace)`.
Reusable for every pair of matrices with the same partitions as `C` and `A`. The
blocks are updated on `ntasks` parallel tasks; `ntasks=1` runs on the calling
task.
"""
struct AdditionWorkspace{T,VC,VA,R}
    c::VC
    a::VA
    workers::Vector{_Worker{T,R}}
end

function AdditionWorkspace(
    C::HMatrixCore, A::HMatrixCore; ntasks::Integer=Threads.nthreads()
)
    T = _checkeltypes(C, A)
    workers = [_Worker{T}() for _ in 1:_checkntasks(ntasks)]
    return AdditionWorkspace(_TreeView(C), _TreeView(A), workers)
end

"""
    axpy!(α, A::HMatrixCore, C::HMatrixCore[, workspace::AdditionWorkspace])
    axpby!(α, A::HMatrixCore, β, C::HMatrixCore[, workspace])

Overwrite `C` with `α*A + C` (or `α*A + β*C`). Both matrices need the same
cluster trees and block partition. Dense blocks are added exactly; every
low-rank block is recompressed once to the larger tolerance of `A` and `C`.
`A` is not modified; `A === C` is allowed.
"""
function LinearAlgebra.axpy!(
    alpha::Number, A::HMatrixCore, C::HMatrixCore, ws::AdditionWorkspace
)
    _checkeltypes(C, A)
    _bind!(ws.c, C)
    _bind!(ws.a, A)
    _hierarchicaladd!(ws.c, ws.a, alpha, ws.workers, _options(C, A))
    return C
end
LinearAlgebra.axpy!(alpha::Number, A::HMatrixCore, C::HMatrixCore) =
    axpy!(alpha, A, C, AdditionWorkspace(C, A))

function LinearAlgebra.axpby!(
    alpha::Number, A::HMatrixCore, beta::Number, C::HMatrixCore, ws::AdditionWorkspace
)
    A === C && return rmul!(C, alpha + beta)
    _checkeltypes(C, A)
    _bind!(ws.c, C)
    _bind!(ws.a, A)
    _checkadditionpartition(ws.c, ws.a)
    rmul!(C, beta)
    _hierarchicaladd!(ws.c, ws.a, alpha, ws.workers, _options(C, A))
    return C
end
LinearAlgebra.axpby!(alpha::Number, A::HMatrixCore, beta::Number, C::HMatrixCore) =
    axpby!(alpha, A, beta, C, AdditionWorkspace(C, A))

"""
    A + B
    A - B

Sum of two H-matrices with the same cluster trees and block partition, as a new
`HMatrixCore` with the larger tolerance of the two.
"""
function Base.:+(A::HMatrixCore, B::HMatrixCore)
    C = _withtol(copy(A), max(A.tol, B.tol))
    return axpy!(true, B, C)
end
function Base.:-(A::HMatrixCore, B::HMatrixCore)
    C = _withtol(copy(A), max(A.tol, B.tol))
    return axpy!(-1, B, C)
end

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

function _zerolike(H::HMatrixCore{T}, tol) where {T}
    near = [zeros(T, size(D)) for D in H.near]
    far = [
        LowRankMatrix{T}(Matrix{T}(undef, size(R, 1), 0), Matrix{T}(undef, 0, size(R, 2)))
        for R in H.far
    ]
    return HMatrix{T}(H.tree, near, far, H.csr, H.csc, H.ordering, H.dim, tol)
end

"""
    A * B

Product of two H-matrices as a new `HMatrixCore` with the block partition of `A`
and the larger tolerance of the two.
"""
function Base.:*(A::HMatrixCore, B::HMatrixCore)
    _checkeltypes(A, B)
    size(A, 2) == size(B, 1) || throw(DimensionMismatch("incompatible matrix sizes"))
    C = _zerolike(A, max(A.tol, B.tol))
    return mul!(C, A, B, true, false)
end

"""
    RecompressionWorkspace(H::HMatrixCore; ntasks=Threads.nthreads())

Buffers for `recompress!(H, workspace)`. The blocks are recompressed on `ntasks`
parallel tasks; `ntasks=1` runs on the calling task.
"""
struct RecompressionWorkspace{T,S}
    scratches::Channel{BlockScratch{T}}
    nchunks::Int
    scheduler::S
end

# One scratch per task: a chunk takes a free scratch and returns it when done.
function RecompressionWorkspace(
    H::HMatrixCore{T}; ntasks::Integer=Threads.nthreads()
) where {T}
    ntasks = _checkntasks(ntasks)
    scratches = Channel{BlockScratch{T}}(ntasks)
    foreach(_ -> put!(scratches, BlockScratch{T}()), 1:ntasks)
    return RecompressionWorkspace(scratches, _nchunks(ntasks), _scheduler(ntasks))
end

"""
    recompress!(H::HMatrixCore[, workspace::RecompressionWorkspace])

Recompress every low-rank block of `H` in place to the smallest rank meeting
`H.tol` in the spectral norm (relative to the block). Assembly by ACA does not
produce optimal ranks, so this usually reduces storage and the cost of later
products and factorizations. The partition does not change.
"""
function recompress!(H::HMatrixCore{T}, ws::RecompressionWorkspace{T}) where {T}
    costs = map(R -> Float64(rank(R))^2 * sum(size(R)), H.far)
    ptr = _costchunks(costs, ws.nchunks)
    tol = H.tol
    tforeach(1:(length(ptr) - 1); scheduler=ws.scheduler) do c
        scratch = take!(ws.scratches)
        try
            for i in ptr[c]:(ptr[c + 1] - 1)
                _recompress!(H.far[i], scratch; atol=0, rtol=tol)
            end
        finally
            put!(ws.scratches, scratch)
        end
    end
    return H
end

function recompress!(H::HMatrixCore)
    @warn "recompress! overwrites the low-rank blocks of the HMatrix." maxlog = 1
    return recompress!(H, RecompressionWorkspace(H))
end
