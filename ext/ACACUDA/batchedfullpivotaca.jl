const _ACA_MAXROWS = 1024
const _ACA_NT = 256   # threads per block; MUST be a power of two and match the launch

# GPU dispatch of the fully-pivoted ACA callable: forwards to batched_fullpivot_aca!.
function (aca::AdaptiveCrossApproximation.ACA{RP,CP,C})(
    buffer::CUDA.AnyCuMatrix{T},
    rowoffsets::AbstractVector{Int},
    ncols::AbstractVector{Int},
    maxrank::Int,
) where {
    T,
    RP<:AdaptiveCrossApproximation.FullPivoting,
    CP<:AdaptiveCrossApproximation.FullPivoting,
    C<:AdaptiveCrossApproximation.FNormEstimator,
}
    return batched_fullpivot_aca!(
        buffer, rowoffsets, ncols, Int(maxrank), real(T)(aca.convergence.tol)
    )
end

"""
    batched_fullpivot_aca!(buffer, rowoffsets, ncols, maxrank, tol) -> ranks, rowpivots, colpivots

Batched fully-pivoted ACA over a flat, column-padded buffer holding multiple blocks
as contiguous row-slices — one CUDA thread-block per matrix block, all launched
together.

Reproduces the plain `ACA{FullPivoting,FullPivoting,FNormEstimator}` callable's
single-block CPU method (`src/fullpivotedaca.jl`) bit-for-bit, including its
deterministic tie-break: among trailing-submatrix entries of equal magnitude, the
smallest column-major linear index wins.

# Arguments

  - `buffer::CUDA.AnyCuMatrix{T}`: `(Σ block-rows) × maxrank` matrix (column-major),
    overwritten in place with Schur complements. Block `i` occupies rows
    `rowoffsets[i]+1 : rowoffsets[i+1]` and columns `1:ncols[i]` (`ncols[i] ≤ maxrank`); columns `ncols[i]+1 : maxrank` are padding and never read.
  - `rowoffsets::AbstractVector{<:Integer}`: length `nblocks+1`, cumulative row
    offsets (`rowoffsets[1] == 0`, `rowoffsets[end] == size(buffer, 1)`).
  - `ncols::AbstractVector{<:Integer}`: length `nblocks`, each block's true column
    count.
  - `maxrank::Int`: buffer column width and per-block rank cap.
  - `tol::Real`: relative Frobenius-norm convergence tolerance.

# Returns

  - `ranks::CuVector{Int32}`: selected rank per block (`0` if a block is all zero).
  - `rowpivots::CuMatrix{Int32}`, `colpivots::CuMatrix{Int32}`: `(maxrank, nblocks)`,
    block-local (1-based) row/column pivot indices in selection order, padded to
    `maxrank` rows per block.
"""
function batched_fullpivot_aca!(
    buffer::CUDA.AnyCuMatrix{T},
    rowoffsets::AbstractVector{<:Integer},
    ncols::AbstractVector{<:Integer},
    maxrank::Int,
    tol::Real,
) where {T}
    nblocks = length(ncols)
    length(rowoffsets) == nblocks + 1 ||
        throw(ArgumentError("rowoffsets must have length nblocks+1"))

    ranks = CUDA.zeros(Int32, nblocks)
    rowpivots = CUDA.zeros(Int32, maxrank, nblocks)
    colpivots = CUDA.zeros(Int32, maxrank, nblocks)
    nblocks == 0 && return ranks, rowpivots, colpivots

    rowoffsets_d = CuArray(Int32.(rowoffsets))
    ncols_d = CuArray(Int32.(ncols))
    tol² = real(T)(tol)^2

    threads = _ACA_NT
    @cuda threads = threads blocks = nblocks _fullpivot_aca_block_kernel!(
        buffer, rowoffsets_d, ncols_d, Int32(maxrank), tol², ranks, rowpivots, colpivots
    )

    return ranks, rowpivots, colpivots
end

# One thread-block per matrix block `bi = blockIdx().x`, threads cooperating over
# that block's (m × n) submatrix. Tie-break MUST match `fullpivotedaca.jl`'s CPU
# loop (`for j in k:n, i in k:m` with strict `>`): among trailing-submatrix entries
# of equal magnitude, the smallest column-major linear index wins.
function _fullpivot_aca_block_kernel!(
    buffer, rowoffsets, ncols, maxrank, tol², ranks, rowpivots, colpivots
)
    bi = blockIdx().x
    tid = threadIdx().x
    nt = blockDim().x
    T = eltype(buffer)
    R = real(T)

    off = rowoffsets[bi]
    m = rowoffsets[bi + 1] - off
    n = ncols[bi]

    rowperm = CuStaticSharedArray(Int32, _ACA_MAXROWS)
    sval = CuStaticSharedArray(R, _ACA_NT)
    sidx = CuStaticSharedArray(Int32, _ACA_NT)
    snorm0 = CuStaticSharedArray(R, 1)

    for i in tid:nt:m
        rowperm[i] = i
    end
    for j in tid:nt:n
        colpivots[j, bi] = j
    end

    local_s = zero(R)
    total = m * n
    for p in (tid - Int32(1)):nt:(total - Int32(1))
        jj = p ÷ m
        ii = p - jj * m
        local_s += abs2(buffer[off + ii + 1, jj + 1])
    end
    sval[tid] = local_s
    sync_threads()
    stride = nt ÷ Int32(2)
    while stride >= Int32(1)
        tid <= stride && (sval[tid] += sval[tid + stride])
        sync_threads()
        stride ÷= Int32(2)
    end
    tid == Int32(1) && (snorm0[1] = sval[1])
    sync_threads()

    rank = Int32(0)
    kmax = min(maxrank, Int32(m), Int32(n))
    for k in Int32(1):kmax
        tm = m - k + Int32(1)
        tn = n - k + Int32(1)
        ntrail = tm * tn
        tbest = zero(R)
        tlin = typemax(Int32)
        for p in (tid - Int32(1)):nt:(ntrail - Int32(1))
            jj = p ÷ tm
            ii = p - jj * tm
            gi = k + ii
            gj = k + jj
            v = abs2(buffer[off + gi, gj])
            lin = (gj - Int32(1)) * m + gi
            if v > tbest || (v == tbest && lin < tlin)
                tbest = v
                tlin = lin
            end
        end
        sval[tid] = tbest
        sidx[tid] = tlin
        sync_threads()
        stride = nt ÷ Int32(2)
        while stride >= Int32(1)
            if tid <= stride
                b = tid + stride
                if sval[b] > sval[tid] || (sval[b] == sval[tid] && sidx[b] < sidx[tid])
                    sval[tid] = sval[b]
                    sidx[tid] = sidx[b]
                end
            end
            sync_threads()
            stride ÷= Int32(2)
        end
        bestval = sval[1]
        bestlin = sidx[1]
        bestval == zero(R) && break
        pc = (bestlin - Int32(1)) ÷ m + Int32(1)
        pr = bestlin - (pc - Int32(1)) * m

        if pr != k
            for j in tid:nt:n
                tmp = buffer[off + k, j]
                buffer[off + k, j] = buffer[off + pr, j]
                buffer[off + pr, j] = tmp
            end
            tid == Int32(1) && ((rowperm[k], rowperm[pr]) = (rowperm[pr], rowperm[k]))
        end
        sync_threads()
        if pc != k
            for i in tid:nt:m
                tmp = buffer[off + i, k]
                buffer[off + i, k] = buffer[off + i, pc]
                buffer[off + i, pc] = tmp
            end
            tid == Int32(1) && (
                (colpivots[k, bi], colpivots[pc, bi]) = (
                    colpivots[pc, bi], colpivots[k, bi]
                )
            )
        end
        sync_threads()

        pinv = inv(buffer[off + k, k])
        um = m - k
        un = n - k
        nupd = um * un
        local_s = zero(R)
        for p in (tid - Int32(1)):nt:(nupd - Int32(1))
            jj = p ÷ um
            ii = p - jj * um
            gi = k + Int32(1) + ii
            gj = k + Int32(1) + jj
            scale = buffer[off + k, gj] * pinv
            val = buffer[off + gi, gj] - buffer[off + gi, k] * scale
            buffer[off + gi, gj] = val
            local_s += abs2(val)
        end
        sval[tid] = local_s
        sync_threads()
        stride = nt ÷ Int32(2)
        while stride >= Int32(1)
            tid <= stride && (sval[tid] += sval[tid + stride])
            sync_threads()
            stride ÷= Int32(2)
        end

        rank = k
        sval[1] <= tol² * snorm0[1] && break
    end

    if tid == Int32(1)
        ranks[bi] = rank
        for a in Int32(1):rank
            rowpivots[a, bi] = rowperm[a]
        end
    end
    return nothing
end

"""
    batched_fullpivot_aca_reference!(aca, buffer, rowoffsets, ncols, maxrank) -> ranks, rowpivots, colpivots

Host CPU reference over the same flat layout as [`batched_fullpivot_aca!`](@ref):
runs the single-block fully-pivoted ACA method (`src/fullpivotedaca.jl`) on each
block's row-slice view, so its output is the ground truth the GPU kernel is checked
against (bit-for-bit pivots/ranks). Overwrites `buffer` in place, exactly like the
GPU path, so callers pass a working copy.
"""
function batched_fullpivot_aca_reference!(
    aca::AdaptiveCrossApproximation.ACA{RP,CP,C},
    buffer::AbstractMatrix{T},
    rowoffsets::AbstractVector{<:Integer},
    ncols::AbstractVector{<:Integer},
    maxrank::Int,
) where {
    T,
    RP<:AdaptiveCrossApproximation.FullPivoting,
    CP<:AdaptiveCrossApproximation.FullPivoting,
    C<:AdaptiveCrossApproximation.FNormEstimator,
}
    nblocks = length(ncols)
    ranks = zeros(Int, nblocks)
    rowpivots = zeros(Int, maxrank, nblocks)
    colpivots = zeros(Int, maxrank, nblocks)
    for i in 1:nblocks
        rows = (rowoffsets[i] + 1):rowoffsets[i + 1]
        A = view(buffer, rows, 1:ncols[i])
        r, rp, cp = aca(A, maxrank)
        ranks[i] = r
        @views rowpivots[1:r, i] .= rp
        @views colpivots[1:r, i] .= cp
    end
    return ranks, rowpivots, colpivots
end
