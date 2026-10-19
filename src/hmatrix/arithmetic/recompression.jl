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

# Scratch views for the QR/LQ/SVD recompression of an m × n block of rank r. A
# short side needs storage only when `packed` (the addition copies its inputs);
# single-block recompression reads that factor directly.
function _recompressionbuffers!(
    scratch::BlockScratch, m::Int, n::Int, r::Int; packed::Bool=false
)
    leftsize, rightsize = checked_mul(m, r), checked_mul(n, r)
    smallsize = checked_mul(min(m, r), min(n, r))
    qrleft, lqright = r < m, r < n
    leftneeded, rightneeded = packed || qrleft, packed || lqright
    _growscratch!(scratch.left, leftneeded ? leftsize : 0)
    _growscratch!(scratch.right, rightneeded ? rightsize : 0)
    _growscratch!(scratch.small, smallsize)
    _growscratch!(scratch.tauleft, qrleft ? r : 0)
    _growscratch!(scratch.tauright, lqright ? r : 0)
    return (
        left=reshape(
            view(scratch.left, 1:(leftneeded ? leftsize : 0)), leftneeded ? m : 0, r
        ),
        right=reshape(
            view(scratch.right, 1:(rightneeded ? rightsize : 0)), r, rightneeded ? n : 0
        ),
        small=reshape(view(scratch.small, 1:smallsize), min(m, r), min(n, r)),
        tauleft=view(scratch.tauleft, 1:(qrleft ? r : 0)),
        tauright=view(scratch.tauright, 1:(lqright ? r : 0)),
    )
end

"""
    _recompressionrank(s, atol, rtol, maxrank; residual=0)

Truncation rank for the singular values `s`: the smallest `k` with
`s[k + 1] + residual <= max(atol, rtol*s[1])`, capped at `maxrank`. `residual`
bounds an error committed before this SVD, such as the range-finder error of
[`_sketchrecompression!`](@ref); it is charged against the tolerance.

Returns `(; rank, error, tolerance, converged, rank_limited)`, where `error` is
the spectral error of the truncated block and `rank_limited` marks a cap by
`maxrank`. Every recompression of the arithmetic returns this information.
"""
function _recompressionrank(s, atol, rtol, maxrank; residual=zero(eltype(s)))
    R = eltype(s)
    tolerance = max(atol, rtol * (isempty(s) ? zero(R) : first(s)))
    required = 0
    for j in length(s):-1:1
        if s[j] + residual > tolerance
            required = j
            break
        end
    end
    k = Int(min(required, maxrank))
    error = (k < length(s) ? s[k + 1] : zero(R)) + residual
    return (
        rank=k,
        error=error,
        tolerance=tolerance,
        converged=error <= tolerance,
        rank_limited=k < required,
    )
end

# Core of the reduced SVD: QR of the left and LQ of the right factor, both in
# place in the buffers, and the product of their triangles. U/V supply a short
# side that needs no factorization; they may be the buffers themselves.
function _preparedrecompressioncore!(buffers, U, V)
    A, B, K = buffers.left, buffers.right, buffers.small
    r = size(U, 2)
    qrleft, lqright = r < size(U, 1), r < size(V, 2)
    if qrleft
        LAPACK.geqrf!(A, buffers.tauleft)
    end
    if lqright
        LAPACK.gelqf!(B, buffers.tauright)
    end
    if qrleft && lqright
        # Copy only R, preserving the reflectors below its diagonal in A.
        copyto!(K, UpperTriangular(view(A, 1:r, :)))
        rmul!(K, LowerTriangular(view(B, :, 1:r)))
    elseif qrleft
        mul!(K, UpperTriangular(view(A, 1:r, :)), V)
    elseif lqright
        mul!(K, U, LowerTriangular(view(B, :, 1:r)))
    else
        mul!(K, U, V)
    end
    return K
end

# Truncated SVD of the prepared core, mapped back through the QR/LQ reflectors.
function _finishrecompression(buffers, m, n, r, atol, rtol, maxrank)
    T = eltype(buffers.small)
    Uk, S, Vt = LAPACK.gesdd!('S', buffers.small)
    info = _recompressionrank(S, atol, rtol, maxrank)
    k = info.rank
    U, V = zeros(T, m, k), zeros(T, k, n)
    for j in 1:k, i in axes(Uk, 1)
        U[i, j] = Uk[i, j] * S[j]
    end
    copyto!(view(V, :, 1:size(Vt, 2)), view(Vt, 1:k, :))
    if k > 0
        r < m && LAPACK.ormqr!('L', 'N', buffers.left, buffers.tauleft, U)
        r < n && LAPACK.ormlq!('R', 'N', buffers.right, buffers.tauright, V)
    end
    return U, V, info
end

# Implicit operator D + U*V + X*Y for the randomized range finder, with an
# optional dense term D (`nothing` when absent).
struct _FactoredSum{E,A,B,C,F}
    D::E
    U::A
    V::B
    X::C
    Y::F
end
_FactoredSum(U, V, X, Y) = _FactoredSum(nothing, U, V, X, Y)
_sketchrank(S::_FactoredSum) = size(S.U, 2) + size(S.X, 2)

function _sketchapply!(W, S::_FactoredSum, Ω, scratch)
    rc, r = size(S.U, 2), _sketchrank(S)
    Z = reshape(view(scratch.small, 1:(r * size(Ω, 2))), r, size(Ω, 2))
    mul!(view(Z, 1:rc, :), S.V, Ω)
    mul!(view(Z, (rc + 1):r, :), S.Y, Ω)
    if S.D === nothing
        mul!(W, S.U, view(Z, 1:rc, :))
    else
        mul!(W, S.D, Ω)
        mul!(W, S.U, view(Z, 1:rc, :), true, true)
    end
    mul!(W, S.X, view(Z, (rc + 1):r, :), true, true)
    return W
end
function _sketchproject!(B, S::_FactoredSum, Q, scratch)
    rc, r = size(S.U, 2), _sketchrank(S)
    Z = reshape(view(scratch.small, 1:(size(Q, 2) * r)), size(Q, 2), r)
    mul!(view(Z, :, 1:rc), adjoint(Q), S.U)
    mul!(view(Z, :, (rc + 1):r), adjoint(Q), S.X)
    if S.D === nothing
        mul!(B, view(Z, :, 1:rc), S.V)
    else
        mul!(B, adjoint(Q), S.D)
        mul!(B, view(Z, :, 1:rc), S.V, true, true)
    end
    mul!(B, view(Z, :, (rc + 1):r), S.Y, true, true)
    return B
end

# Sample columns are far from overflow; avoid the scaled generic 2-norm.
_sketchnorm(x) = sqrt(real(dot(x, x)))

# Orthogonalize column c of A against A[:, 1:k] by classical Gram-Schmidt with
# one reorthogonalization; returns the norms before and after. Level-1 BLAS on
# purpose: concurrent OpenBLAS gemv calls serialize on a global buffer lock.
function _sketchorthogonalize!(A, c, k, h)
    w = view(A, :, c)
    before = _sketchnorm(w)
    for _ in 1:(k > 0 ? 2 : 0)
        for j in 1:k
            h[j] = dot(view(A, :, j), w)
        end
        for j in 1:k
            axpy!(-h[j], view(A, :, j), w)
        end
    end
    return before, _sketchnorm(w)
end

# Append the already orthogonalized column c as basis column k + 1 unless it is
# numerically contained in the basis. Returns the new basis size.
function _sketchappend!(A, c, k, before, after)
    after > 64 * eps(typeof(after)) * before || return k
    k += 1
    c == k || copyto!(view(A, :, k), view(A, :, c))
    rmul!(view(A, :, k), inv(after))
    return k
end

"""
    _sketchrecompression!(S, m, n, guess, qcap, scratch, atol, rtol, maxrank)

Randomized spectral-norm recompression of an implicit `m × n` operator `S`
(adaptive randomized range finder of Halko, Martinsson & Tropp 2011, Alg. 4.2,
with blocked sampling). `S` only needs to apply itself to a block of vectors
(`_sketchapply!`) and to project onto a basis (`_sketchproject!`), so sums of
factors and hierarchical products are never formed explicitly.

The first `guess` samples build an orthonormal basis `Q`. Each further block of
`p = 8` Gaussian samples, projected onto the complement of `Q`, certifies

    ‖(I - QQ')S‖ ≤ 10 √(2/π) maxᵢ ‖(I - QQ')S ωᵢ‖

except with probability `10^-p`, or is appended to `Q`. The bound is driven
below a tenth of the tolerance and charged against it after the SVD of `Q'S`
(see [`_recompressionrank`](@ref)). The sampling RNG has a fixed seed, so
results are reproducible.

Returns `(U, V, info)`, or `nothing` if `Q` would exceed `qcap` columns; the
caller then falls back to a deterministic QR/LQ/SVD recompression.
"""
function _sketchrecompression!(S, m, n, guess, qcap, scratch, atol, rtol, maxrank)
    p = 8
    width = qcap + p
    guess = min(guess, qcap)
    T = eltype(scratch.left)
    R = real(T)
    _growscratch!(scratch.left, checked_mul(m, width))
    _growscratch!(scratch.right, checked_mul(n, width))
    _growscratch!(scratch.small, checked_mul(max(_sketchrank(S), width), width))
    _growscratch!(scratch.tauleft, width)
    _growscratch!(scratch.tauright, width)
    A = reshape(view(scratch.left, 1:(m * width)), m, width)
    rng = Random.Xoshiro(
        0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1
    )
    k, sigma, bound, built = 0, zero(R), zero(R), guess
    while true
        b = built + p
        k + b <= width || return nothing
        Ω = reshape(view(scratch.right, 1:(n * b)), n, b)
        Random.randn!(rng, Ω)
        W = view(A, :, (k + 1):(k + b))
        _sketchapply!(W, S, Ω, scratch)
        # ||S w|| / ||w|| <= sigma_1 bounds the spectral tolerance from below.
        for j in 1:b
            sigma = max(sigma, _sketchnorm(view(W, :, j)) / _sketchnorm(view(Ω, :, j)))
        end
        first, tests = k + 1, k + built + 1
        for c in first:(tests - 1)
            before, after = _sketchorthogonalize!(A, c, k, scratch.tauleft)
            k = _sketchappend!(A, c, k, before, after)
        end
        # Move tests directly behind the basis; they are projected afterwards.
        offset = tests - (k + 1)
        if offset > 0
            for j in 0:(p - 1)
                copyto!(view(A, :, k + 1 + j), view(A, :, tests + j))
            end
        end
        largest = zero(R)
        for c in (k + 1):(k + p)
            _, after = _sketchorthogonalize!(A, c, k, scratch.tauleft)
            largest = max(largest, after)
        end
        bound = 10 * sqrt(2 / R(pi)) * largest
        bound <= max(atol, rtol * sigma) / 10 && break
        # Tests failed: they are valid samples of the complement, so keep them.
        base = k
        for c in (base + 1):(base + p)
            before, after = _sketchorthogonalize!(A, c, k, scratch.tauleft)
            k = _sketchappend!(A, c, k, before, after)
        end
        # Grow geometrically, but only into the remaining capacity.
        built = max(p, min(k, width - k - 2p))
    end
    Q = view(A, :, 1:k)
    B = reshape(view(scratch.right, 1:(k * n)), k, n)
    tau = view(scratch.tauright, 1:k)
    U, V = Matrix{T}(undef, m, 0), Matrix{T}(undef, 0, n)
    if k > 0
        _sketchproject!(B, S, Q, scratch)
        # The SVD of the short, wide B = Q'S needs only the triangle of a QR of
        # its tall adjoint B' = Q_B*R; the truncation Ul_r*Ul_r'*B is then two
        # GEMMs with the intact B.
        _growscratch!(scratch.small, checked_mul(k, n + k))
        Bt = reshape(view(scratch.small, 1:(n * k)), n, k)
        adjoint!(Bt, B)
        LAPACK.geqrf!(Bt, tau)
        L = reshape(view(scratch.small, (k * n + 1):(k * n + k * k)), k, k)
        for j in 1:k, i in 1:k
            L[i, j] = i >= j ? conj(Bt[j, i]) : zero(T)
        end
        Ul, singular = LAPACK.gesdd!('S', L)
        info = _recompressionrank(singular, atol, rtol, maxrank; residual=bound)
        r = info.rank
        if r > 0
            Ur = view(Ul, :, 1:r)
            U, V = Matrix{T}(undef, m, r), Matrix{T}(undef, r, n)
            mul!(U, Q, Ur)
            mul!(V, adjoint(Ur), B)
        end
    else
        info = _recompressionrank(R[], atol, rtol, maxrank; residual=bound)
    end
    return U, V, info
end

"""
    _flushfactors!(C::LowRankMatrix, X, Y, scratch; atol, rtol, maxrank, α=1, sketch=false, guess)

Replace `C` by a recompression of `C + α*X*Y` in one pass. The deterministic path
stacks the factors, `[C.U X] * [C.V; Y]`, and truncates them by QR of the left
factor, LQ of the right factor and an SVD of the small core; when the stacked
rank reaches both dimensions, the dense sum is truncated instead. With
`sketch=true`, wide updates first try [`_sketchrecompression!`](@ref), whose
cost is linear in the stacked rank; `guess` is its first sample count. Returns
the truncation information of [`_recompressionrank`](@ref).
"""
function _flushfactors!(
    C::LowRankMatrix{T},
    X,
    Y,
    scratch;
    atol,
    rtol,
    maxrank,
    alpha=one(T),
    sketch=false,
    guess=rank(C) + 5,
) where {T}
    ra = size(X, 2)
    rc = rank(C)
    r = checked_add(rc, ra)
    m, n = size(C)
    densecore = r >= m && r >= n
    if sketch && isone(alpha) && !densecore
        S = _FactoredSum(C.U, C.V, X, Y)
        # At small input rank the sketch beats QR/LQ/SVD even near full rank,
        # and the basis can always reach the exact r-dimensional range.
        qcap = r <= 64 ? min(m, n, r) : min(m, n, r) ÷ 2
        result = _sketchrecompression!(S, m, n, guess, qcap, scratch, atol, rtol, maxrank)
        if !isnothing(result)
            C.U, C.V = result[1], result[2]
            return result[3]
        end
    end
    buffers = _recompressionbuffers!(scratch, m, n, r; packed=(!densecore))
    if densecore
        mul!(buffers.small, C.U, C.V)
        mul!(buffers.small, X, Y, alpha, one(T))
    else
        copyto!(view(buffers.left, :, 1:rc), C.U)
        copyto!(view(buffers.right, 1:rc, :), C.V)
        copyto!(view(buffers.left, :, (rc + 1):r), X)
        copyto!(view(buffers.right, (rc + 1):r, :), Y)
        # Scale the smaller factor.
        if !isone(alpha)
            if m <= n
                rmul!(view(buffers.left, :, (rc + 1):r), alpha)
            else
                rmul!(view(buffers.right, (rc + 1):r, :), alpha)
            end
        end
        _preparedrecompressioncore!(buffers, buffers.left, buffers.right)
    end
    U, V, info = _finishrecompression(buffers, m, n, r, atol, rtol, maxrank)
    C.U, C.V = U, V
    return info
end

"""
    _recompress!(R::LowRankMatrix, scratch; atol=0, rtol=1e-6, maxrank=typemax(Int))

Replace the factors of `R = U*V` by a truncated reduced SVD: QR of `U`, LQ of
`V`, SVD of the product of the two triangles. The new rank is the smallest one
whose largest discarded singular value is at most `max(atol, rtol*opnorm(R))`,
capped at `maxrank`. The new factors own their storage. Returns the truncation
information of [`_recompressionrank`](@ref).
"""
function _recompress!(
    R::LowRankMatrix{T},
    scratch::BlockScratch{T};
    atol::Real=0,
    rtol::Real=1e-6,
    maxrank::Integer=typemax(Int),
) where {T<:LinearAlgebra.BlasFloat}
    m, n = size(R)
    r = rank(R)
    if min(m, n, r) == 0
        info = _recompressionrank(real(T)[], atol, rtol, maxrank)
        U, V = Matrix{T}(undef, m, 0), Matrix{T}(undef, 0, n)
    else
        buffers = _recompressionbuffers!(scratch, m, n, r)
        r < m && copyto!(buffers.left, R.U)
        r < n && copyto!(buffers.right, R.V)
        _preparedrecompressioncore!(buffers, R.U, R.V)
        U, V, info = _finishrecompression(buffers, m, n, r, atol, rtol, maxrank)
    end
    R.U, R.V = U, V
    return info
end
