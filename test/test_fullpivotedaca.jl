using AdaptiveCrossApproximation
using CUDA
using LinearAlgebra
using Random
using Test

# Pack a single dense matrix into the batched flat-buffer contract (nb == 1).
function _pack_single_block(A::AbstractMatrix)
    m, n = size(A)
    return copy(A), [0, m], [n]
end

@testset "fullpivotedaca.jl" begin
    @testset "ACA (FullPivoting)" begin
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
end
