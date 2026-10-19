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

# Exact scaling of the smaller factor; zero leaves a rank-zero block.
function _scalelowrank!(R::LowRankMatrix{T}, alpha) where {T}
    m, n = size(R)
    if iszero(alpha)
        R.U, R.V = Matrix{T}(undef, m, 0), Matrix{T}(undef, 0, n)
    else
        m <= n ? rmul!(R.U, alpha) : rmul!(R.V, alpha)
    end
    return R
end

"""
    _addlowrank!(C::LowRankMatrix, A::LowRankMatrix, α, scratch; atol=0, rtol=1e-6, maxrank)

Replace `C` by a recompressed approximation of `C + α*A`, with tolerances
relative to the sum. The factors are stacked, `[C.U α*A.U] * [C.V; A.V]`, and
truncated by [`_flushfactors!`](@ref). Returns the truncation information of
[`_recompressionrank`](@ref).
"""
function _addlowrank!(
    C::LowRankMatrix{T},
    A::LowRankMatrix{T},
    alpha::Number,
    scratch::BlockScratch{T};
    atol::Real=0,
    rtol::Real=1e-6,
    maxrank::Integer=typemax(Int),
) where {T<:LinearAlgebra.BlasFloat}
    size(C) == size(A) || throw(DimensionMismatch("low-rank block dimensions differ"))
    alpha = convert(T, alpha)
    if iszero(alpha) || min(rank(A), size(C)...) == 0
        return _recompress!(C, scratch; atol, rtol, maxrank)
    end
    return _flushfactors!(C, A.U, A.V, scratch; atol, rtol, maxrank, alpha)
end

# Whether two views are bound to the same blocks.
_bound(C::_TreeView, A::_TreeView) =
    length(C.near) == length(A.near) &&
    length(C.far) == length(A.far) &&
    all(i -> C.near[i] === A.near[i], eachindex(C.near)) &&
    all(i -> C.far[i] === A.far[i], eachindex(C.far))

function _checkadditionpartition(C, A)
    size(C) == size(A) || throw(DimensionMismatch("matrix dimensions differ"))
    testtree(C.tree) === testtree(A.tree) && trialtree(C.tree) === trialtree(A.tree) ||
        throw(ArgumentError("addition requires the same row and column cluster trees"))
    c, a = C.csr, A.csr
    c === a ||
        c.rowptr == a.rowptr && c.colnodes == a.colnodes && c.isfar == a.isfar ||
        throw(
            ArgumentError("addition requires identical terminal partitions and block kinds")
        )
    return nothing
end

"""
    _hierarchicaladd!(C::_TreeView, A::_TreeView, α, workers, options)

`C += α*A` for views with identical trees and partitions; their storage orders
may differ. Dense blocks are updated exactly and every far block is
recompressed once (see [`_addlowrank!`](@ref)). The blocks are independent and
distributed over `workers`. `C += α*C` is an exact scaling. Returns the
aggregated truncation status.
"""
function _hierarchicaladd!(
    C::_TreeView{T}, A::_TreeView{T}, coefficient, workers::AbstractVector, options
) where {T}
    # A fresh name: closures box reassigned captured variables.
    alpha = convert(T, coefficient)
    isfinite(alpha) || throw(ArgumentError("alpha must be finite"))
    _checkadditionpartition(C, A)
    _resetstatus!(workers)
    if _bound(C, A)
        # C += alpha*C is an exact scaling.
        beta = one(T) + alpha
        isfinite(beta) || throw(ArgumentError("1 + alpha must be finite"))
        foreach(D -> rmul!(D, beta), C.near)
        foreach(R -> _scalelowrank!(R, beta), C.far)
    elseif !iszero(alpha)
        c, a = C.csr, A.csr
        n = length(c.storageidx)
        independent = _TaskGraph(ones(Int, n + 1), Int[], zeros(Int, n))
        _rungraph!(independent, workers) do k, worker
            if c.isfar[k]
                R, S = C.far[c.storageidx[k]], A.far[a.storageidx[k]]
                _record!(
                    worker.status, _addlowrank!(R, S, alpha, worker.scratch; options...), 0
                )
            else
                axpy!(alpha, A.near[a.storageidx[k]], C.near[c.storageidx[k]])
            end
            return true
        end
    end
    return _workerstatus(workers)
end
