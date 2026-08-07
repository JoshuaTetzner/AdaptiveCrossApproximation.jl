using AdaptiveCrossApproximation
using CUDA
using LinearAlgebra
using Random
using Test

# A handful of blocks of differing shape/rank, packed as contiguous row-slices of
# one (Σrows × maxrank) column-padded buffer. Includes a full-column-rank block
# and an all-zero (rank-0) block to exercise the boundary cases.
function _build_level_buffer(; T=ComplexF64, maxrank=30, seed=1)
    Random.seed!(seed)
    specs = [
        (m=120, n=25, r=8),
        (m=64, n=30, r=17),
        (m=200, n=12, r=5),
        (m=90, n=20, r=20),   # full column rank
        (m=50, n=10, r=0),    # all-zero block
    ]
    nb = length(specs)
    rowoffsets = zeros(Int, nb + 1)
    for i in 1:nb
        rowoffsets[i + 1] = rowoffsets[i] + specs[i].m
    end
    ncols = [s.n for s in specs]

    buffer = zeros(T, rowoffsets[end], maxrank)
    originals = Vector{Matrix{T}}(undef, nb)
    for i in 1:nb
        m, n, r = specs[i].m, specs[i].n, specs[i].r
        block = r == 0 ? zeros(T, m, n) : randn(T, m, r) * randn(T, r, n)  # exact rank r
        originals[i] = copy(block)
        buffer[(rowoffsets[i] + 1):rowoffsets[i + 1], 1:n] .= block
    end
    return buffer, originals, rowoffsets, ncols, specs
end

# CUR reconstruction error of block `i` from selected `rank`/pivots.
function _cur_relerr(A, rank, rowpivots, colpivots)
    rank == 0 && return norm(A)  # a nonzero block picked no pivots => full error
    rp = rowpivots[1:rank]
    cp = colpivots[1:rank]
    recon = A[:, cp] * (A[rp, cp] \ A[rp, :])
    return norm(recon - A) / norm(A)
end

# Pack a single dense matrix into the batched flat-buffer contract (nb == 1).
function _pack_single_block(A::AbstractMatrix)
    m, n = size(A)
    return copy(A), [0, m], [n]
end

@testset "Batched fully-pivoted ACA (flat level buffer)" begin
    acaext = Base.get_extension(AdaptiveCrossApproximation, :ACACUDA)
    @test acaext !== nothing

    maxrank = 30
    aca = ACA(FullPivoting(), FullPivoting(), FNormEstimator(1e-12))

    @testset "CPU reference (T=$T)" for T in (ComplexF64, Float64)
        buffer, originals, rowoffsets, ncols, specs = _build_level_buffer(;
            T=T, maxrank=maxrank
        )
        nb = length(specs)

        ranks, rowpivots, colpivots = acaext.batched_fullpivot_aca_reference!(
            aca, copy(buffer), rowoffsets, ncols, maxrank
        )

        for i in 1:nb
            @test ranks[i] == specs[i].r
            err = _cur_relerr(originals[i], ranks[i], rowpivots[:, i], colpivots[:, i])
            @test err < 1e-9
        end
    end

    if CUDA.functional()
        @testset "GPU kernel matches CPU reference (T=$T)" for T in (ComplexF64, Float64)
            buffer, originals, rowoffsets, ncols, specs = _build_level_buffer(;
                T=T, maxrank=maxrank
            )
            nb = length(specs)

            rk, rp, cp = acaext.batched_fullpivot_aca_reference!(
                aca, copy(buffer), rowoffsets, ncols, maxrank
            )

            rk_d, rp_d, cp_d = aca(CuArray(buffer), rowoffsets, ncols, maxrank)
            rk_g = Int.(Array(rk_d))
            rp_g = Int.(Array(rp_d))
            cp_g = Int.(Array(cp_d))

            @test rk_g == rk
            for i in 1:nb
                r = rk[i]
                @test rp_g[1:r, i] == rp[1:r, i]
                @test cp_g[1:r, i] == cp[1:r, i]
                err = _cur_relerr(originals[i], rk_g[i], rp_g[:, i], cp_g[:, i])
                @test err < 1e-9
            end
        end
    else
        @info "No functional CUDA device: skipping GPU batched-ACA kernel checks."
    end
end

# The same kernel/CPU-reference contract, exercised through the single-matrix
# entry points instead of the batched flat-buffer helper above (a full dense
# matrix, GPU run as a 1-block batch via `_pack_single_block`).
@testset "Fully-pivoted ACA on a single full matrix (CPU vs GPU)" begin
    maxrank = 30

    @testset "Exact low-rank matrix, T=$T" for T in (ComplexF64, Float64)
        Random.seed!(3)
        m, n, r = 45, 25, 12  # n <= maxrank, per batched_fullpivot_aca!'s contract
        A = randn(T, m, r) * randn(T, r, n)

        aca = ACA(FullPivoting(), FullPivoting(), FNormEstimator(1e-10))
        rank, rowpivots, colpivots = aca(copy(A), maxrank)
        @test rank == r
        @test _cur_relerr(A, rank, rowpivots, colpivots) < 1e-9

        if CUDA.functional()
            buffer, rowoffsets, ncols = _pack_single_block(A)
            rk_d, rp_d, cp_d = aca(CuArray(buffer), rowoffsets, ncols, maxrank)
            @test Int(Array(rk_d)[1]) == rank
            @test Int.(Array(rp_d)[1:rank, 1]) == rowpivots
            @test Int.(Array(cp_d)[1:rank, 1]) == colpivots
        end
    end

    @testset "Tie-break determinism, T=$T" for T in (ComplexF64, Float64)
        # (1,3) and (4,7) tie at |.|=2; smallest column-major index must win, so
        # pivot 1 lands on (1,3), not (4,7).
        m, n = 8, 10
        A = zeros(T, m, n)
        A[1, 3] = 2
        A[4, 7] = 2
        A[2, 5] = 1

        aca = ACA(FullPivoting(), FullPivoting(), FNormEstimator(1e-10))
        rank, rowpivots, colpivots = aca(copy(A), maxrank)
        @test rowpivots[1] == 1
        @test colpivots[1] == 3

        if CUDA.functional()
            buffer, rowoffsets, ncols = _pack_single_block(A)
            rk_d, rp_d, cp_d = aca(CuArray(buffer), rowoffsets, ncols, maxrank)
            @test Int.(Array(rp_d)[1:rank, 1]) == rowpivots
            @test Int.(Array(cp_d)[1:rank, 1]) == colpivots
        end
    end

    @testset "maxrank cap (CPU only), T=$T" for T in (ComplexF64, Float64)
        # No GPU counterpart: forcing cappedrank < n here would violate
        # batched_fullpivot_aca!'s ncols[i] <= maxrank contract.
        Random.seed!(11)
        m = n = 10
        cappedrank = 4
        A = randn(T, m, n)  # generically full rank (10), so maxrank must bind

        aca = ACA(FullPivoting(), FullPivoting(), FNormEstimator(1e-14))
        rank, rowpivots, colpivots = aca(copy(A), cappedrank)
        @test rank == cappedrank
        @test length(rowpivots) == cappedrank
        @test length(colpivots) == cappedrank
        @test _cur_relerr(A, rank, rowpivots, colpivots) > 1e-10  # genuinely capped
    end
end
