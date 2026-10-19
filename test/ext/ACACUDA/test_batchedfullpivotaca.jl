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

@testset "batchedfullpivotaca.jl" begin
    @testset "batched_fullpivot_aca_reference! / batched_fullpivot_aca!" begin
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
            @testset "GPU kernel matches CPU reference (T=$T)" for T in
                                                                   (ComplexF64, Float64)
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

    @testset "ACA / batched_fullpivot_aca_reference! (sampled residuals)" begin
        acaext = Base.get_extension(AdaptiveCrossApproximation, :ACACUDA)
        for T in (Float32, Float64, ComplexF32, ComplexF64), m in (3, 5)
            A = zeros(T, m, 3)
            A[1, 1], A[2, 2], A[3, 3] = 1, 0.009, 0.001
            tol = real(T)(1e-2)
            aca = ACA(FullPivoting(), FullPivoting(), FNormEstimator(tol))
            @test first(aca(copy(A), 3)) == 1
            rank, rp, cp = aca(copy(A), 3; sampled=true)
            @test rank == 2
            @test rp == collect(1:rank)
            @test cp == collect(1:rank)
            residual = A - A[:, cp] * (A[rp, cp] \ A[rp, :])
            estimated_error = norm(residual) / norm(A) * sqrt(3 / (3-rank))
            @test estimated_error <= tol
            # Applying s/(s-r) to the norm instead of its square is too strict:
            # this sample must stop at rank 1 with the square-root norm correction.
            B = copy(A)
            B[2, 2] = 0.0075
            @test first(aca(copy(B), 3; sampled=true)) == 1
            if CUDA.functional()
                ranks_d, rows_d, cols_d = aca(CuArray(A), [0, m], [3], 3; sampled=true)
                @test Array(ranks_d) == [rank]
                @test Int.(Array(rows_d)[1:rank, 1]) == rp
                @test Int.(Array(cols_d)[1:rank, 1]) == cp
                ranks_d, _, _ = aca(CuArray(B), [0, m], [3], 3; sampled=true)
                @test Array(ranks_d) == [1]
            end
            # Different sampled widths use their own factors, without full counts.
            buffer = zeros(T, 3m, 8)
            for i in 1:3
                buffer[((i - 1) * m + 1):(i * m), 1:3] = A
            end
            offsets = [0, m, 2m, 3m]
            ncols = [3, 4, 8]
            ranks, rows, cols = acaext.batched_fullpivot_aca_reference!(
                aca, copy(buffer), offsets, ncols, 8; sampled=true
            )
            @test ranks == [2, 2, 1]
            if CUDA.functional()
                ranks_d, rows_d, cols_d = aca(
                    CuArray(buffer), offsets, ncols, 8; sampled=true
                )
                @test Int.(Array(ranks_d)) == ranks
                @test Int.(Array(rows_d)) == rows
                gpu_cols = Int.(Array(cols_d))
                for i in eachindex(ranks)
                    @test gpu_cols[1:ranks[i], i] == cols[1:ranks[i], i]
                end
            end
        end

        # Exhausting either dimension terminates without dividing by zero.
        # Zero blocks keep rank zero in both ordinary and sampled modes.
        for A in (reshape([2.0, 1.0], 1, 2), reshape([2.0, 1.0], 2, 1), zeros(3, 2)),
            sampled in (false, true)

            aca = ACA(FullPivoting(), FullPivoting(), FNormEstimator(1e-3))
            rank, rp, cp = aca(copy(A), min(size(A)...); sampled)
            @test rank == (iszero(norm(A)) ? 0 : 1)
            if CUDA.functional()
                ranks_d, _, _ = aca(
                    CuArray(A), [0, size(A, 1)], [size(A, 2)], size(A, 2); sampled
                )
                @test Array(ranks_d) == [rank]
            end
        end
    end
end
