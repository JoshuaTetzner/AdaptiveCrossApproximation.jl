# Storage policy stays behind this helper: the numerical kernel only sees
# matrices and Householder vectors. LQ needs V rather than adjoint(V). Short
# sides need storage only when packing addition inputs; single-block
# recompression reads those factors directly.
function _recompressionbuffers!(
    scratch::BlockScratch, m::Int, n::Int, r::Int; packed::Bool=false
)
    _, _, _, leftsize, rightsize, smallsize = _blockscratchsizes(m, n, r)
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

function _recompressionoptions(atol::Real, rtol::Real, maxrank::Integer)
    isfinite(atol) && atol >= 0 ||
        throw(ArgumentError("atol must be finite and nonnegative"))
    isfinite(rtol) && rtol >= 0 ||
        throw(ArgumentError("rtol must be finite and nonnegative"))
    maxrank >= 0 || throw(ArgumentError("maxrank must be nonnegative"))
    return nothing
end

# Spectral-norm truncation: return the smallest admissible rank, then apply the
# independent hard rank cap. `residual` bounds an error already committed before
# this SVD; it is charged additively against the tolerance and included in `error`.
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

function _recompressioncore!(buffers, U, V)
    r = size(U, 2)
    r < size(U, 1) && copyto!(buffers.left, U)
    r < size(V, 2) && copyto!(buffers.right, V)
    return _preparedrecompressioncore!(buffers, U, V)
end

# Destructively factorize prepared buffers. U/V supply an unfactorized short
# side, if any; they may be the buffers themselves. Never reserve scratch here.
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
        # Both dimensions are <= r: the reduced core is full-block sized, but
        # still lives in scratch. No additional dense U*V matrix is allocated.
        mul!(K, U, V)
    end
    return K
end

# Reduced SVD of the core; Julia's LAPACK wrapper allocates its work arrays.
function _recompresssvd!(K)
    U, S, Vt = LAPACK.gesdd!('S', K)
    return (; U, S, Vt)
end

# Shared by single-block recompression and addition, after preparing the core.
function _finishrecompression(buffers, m, n, r, atol, rtol, maxrank)
    F = _recompresssvd!(buffers.small)
    info = _recompressionrank(F.S, atol, rtol, maxrank)
    U, V = _recompressionfactors(buffers, F, m, n, r, info.rank)
    return U, V, info
end

function _recompressionfactors(buffers, F, m, n, r, k)
    T = eltype(buffers.small)
    U = zeros(T, m, k)
    V = zeros(T, k, n)
    for j in 1:k, i in axes(F.U, 1)
        U[i, j] = F.U[i, j] * F.S[j]
    end
    copyto!(view(V, :, 1:size(F.Vt, 2)), view(F.Vt, 1:k, :))
    if k > 0
        r < m && LAPACK.ormqr!('L', 'N', buffers.left, buffers.tauleft, U)
        r < n && LAPACK.ormlq!('R', 'N', buffers.right, buffers.tauright, V)
    end
    return U, V
end

# Implicit operator D + U*V + X*Y for the randomized range finder below, with an
# optional dense term D (`nothing` when absent). Only S*Ω and Q'*S are required.
# Buffers come from the caller's scratch; operands never alias it.
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

# Fixed seeds keep factorizations reproducible and independent of call history.
_sketchrng() = Random.Xoshiro(
    0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1
)

# Sample columns are far from overflow; avoid the scaled generic 2-norm.
_sketchnorm(x) = sqrt(real(dot(x, x)))

# Orthogonalize column c of A against A[:, 1:k] by classical Gram-Schmidt with
# one reorthogonalization. Returns the norms before and after projection. Level-1
# BLAS: concurrent OpenBLAS matrix-vector calls serialize on a buffer lock.
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

# Randomized spectral-norm recompression of an m × n operator S (Halko, Martinsson
# & Tropp 2011, Alg. 4.2 with blocked sampling). The first `guess` samples build
# the basis Q; each later block of p samples is projected onto the complement of
# Q and certifies ||(I - QQ')S|| <= 10 sqrt(2/pi) max_i ||(I - QQ')S w_i|| except
# with probability 10^-p. Failed tests are appended and sampling continues. The
# bound is driven below a tenth of a lower bound for the truncation tolerance,
# then charged against that tolerance after an SVD of Q'S (via a QR of its
# tall adjoint, since Q'S is short and wide). Returns `nothing`,
# leaving S unchanged, if the basis would exceed `qcap` columns.
function _sketchrecompression!(S, m, n, guess, qcap, scratch, atol, rtol, maxrank)
    p = 8
    width = qcap + p
    guess = min(guess, qcap)
    T = eltype(scratch.left)
    R = real(T)
    checked_mul = Base.Checked.checked_mul
    _growscratch!(scratch.left, checked_mul(m, width))
    _growscratch!(scratch.right, checked_mul(n, width))
    _growscratch!(scratch.small, checked_mul(max(_sketchrank(S), width), width))
    _growscratch!(scratch.tauleft, width)
    _growscratch!(scratch.tauright, width)
    A = reshape(view(scratch.left, 1:(m * width)), m, width)
    rng = _sketchrng()
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
        # B = R'*Q_B' from a QR of the tall, contiguous B' (faster than LQ of the
        # short, wide B). R' only supplies the small SVD; the truncation is then
        # Ul_r * Ul_r' * B, applied with two GEMMs to the intact B instead of
        # Householder reflectors. U = Q*Ul_r keeps orthonormal columns.
        _growscratch!(scratch.small, checked_mul(k, n + k))
        Bt = reshape(view(scratch.small, 1:(n * k)), n, k)
        adjoint!(Bt, B)
        LAPACK.geqrf!(Bt, tau)
        L = reshape(view(scratch.small, (k * n + 1):(k * n + k * k)), k, k)
        for j in 1:k, i in 1:k
            L[i, j] = i >= j ? conj(Bt[j, i]) : zero(T)
        end
        F = _recompresssvd!(L)
        info = _recompressionrank(F.S, atol, rtol, maxrank; residual=bound)
        r = info.rank
        if r > 0
            Ul = view(F.U, :, 1:r)
            U, V = Matrix{T}(undef, m, r), Matrix{T}(undef, r, n)
            mul!(U, Q, Ul)
            mul!(V, adjoint(Ul), B)
        end
    else
        info = _recompressionrank(R[], atol, rtol, maxrank; residual=bound)
    end
    return U, V, info
end

# Recompress C + alpha*X*Y in one pass, for low-rank addition and
# destination-local product buffers. Callers validate aliases and limits before
# borrowing these views. Source factors are never overwritten; the new factors
# are committed only on success.
# With `sketch=true`, wide flushes first try the randomized range finder, whose cost is linear in the pending rank; X and Y must not alias scratch.
# `guess` is the first sample count (default: destination rank plus five).
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
    r = Base.Checked.checked_add(rc, ra)
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

# Replace the factors of R = U*V with a truncated QR/LQ/SVD approximation: the
# smallest rank whose largest discarded singular value is at most
# max(atol, rtol*opnorm(R)), then capped at maxrank. Returns
# (rank, error, tolerance, converged, rank_limited) with the spectral truncation
# error estimated from the reduced SVD. New factors own their storage; old
# factor arrays are not overwritten. Scratch uses O((m+n)r + r^2) storage.
function _recompress!(
    R::LowRankMatrix{T},
    scratch::BlockScratch{T};
    atol::Real=0,
    rtol::Real=1e-6,
    maxrank::Integer=typemax(Int),
) where {T<:LinearAlgebra.BlasFloat}
    _recompressionoptions(atol, rtol, maxrank)
    m, n = size(R)
    r = rank(R)
    size(R.V, 1) == r || throw(DimensionMismatch("incompatible low-rank factors"))
    if min(m, n, r) == 0
        info = _recompressionrank(real(T)[], atol, rtol, maxrank)
        U, V = Matrix{T}(undef, m, 0), Matrix{T}(undef, 0, n)
    else
        buffers = _recompressionbuffers!(scratch, m, n, r)
        _recompressioncore!(buffers, R.U, R.V)
        U, V, info = _finishrecompression(buffers, m, n, r, atol, rtol, maxrank)
    end
    # Commit only after all numerical work succeeds.
    R.U, R.V = U, V
    return info
end
