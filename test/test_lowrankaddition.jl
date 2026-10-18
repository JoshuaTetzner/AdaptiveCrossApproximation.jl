using AdaptiveCrossApproximation
using LinearAlgebra
using Random
using Test

function addition_fixture(rng, T, m, n, r)
    U, V = randn(rng, T, m, r), randn(rng, T, r, n)
    for j in 1:r
        view(U, :, j) .*= 0.35^(j - 1)
    end
    return AdaptiveCrossApproximation.LowRankMatrix{T}(U, V)
end

function addition_allocations(C, A, scratch)
    U, V = C.U, C.V
    AdaptiveCrossApproximation._addlowrank!(C, A, 0.7, scratch)
    C.U, C.V = U, V
    return @allocated AdaptiveCrossApproximation._addlowrank!(C, A, 0.7, scratch)
end

@testset "Low-rank addition" begin
    ACA = AdaptiveCrossApproximation
    rng = MersenneTwister(829)
    shapes = (
        (29, 23, 4, 6),
        (29, 5, 4, 6),
        (5, 29, 4, 6),
        (5, 6, 4, 6),
        (10, 21, 4, 6),
        (21, 10, 4, 6),
        (10, 10, 4, 6),
        (1, 9, 2, 1),
        (9, 1, 1, 2),
    )
    for T in (Float64, ComplexF64, Float32, ComplexF32)
        alpha = T <: Real ? T(-0.7) : T(-0.7 + 0.4im)
        roundoff = 300eps(real(T))
        @testset "$T accuracy and ownership" begin
            scratch = ACA.BlockScratch{T}()
            for (m, n, rc, ra) in shapes, cap in (2, typemax(Int))
                C = addition_fixture(rng, T, m, n, rc)
                A = addition_fixture(rng, T, m, n, ra)
                oldU, oldV, sourceU, sourceV = C.U, C.V, A.U, A.V
                snapshots = copy.((oldU, oldV, sourceU, sourceV))
                expected = Matrix(C) + alpha * Matrix(A)
                F = svd(expected)
                tolerance = max(1e-5, 0.025first(F.S))
                tail(k) = k == length(F.S) ? zero(real(T)) : F.S[k + 1]
                required = findfirst(k -> tail(k) <= tolerance, 0:length(F.S)) - 1
                k = min(required, cap)
                info = @inferred ACA._addlowrank!(
                    C, A, alpha, scratch; atol=1e-5, rtol=0.025, maxrank=cap
                )
                @test rank(C) == info.rank == k
                @test info.rank_limited == (cap < required)
                @test info.converged == (cap >= required)
                @test info.tolerance ≈ tolerance rtol=roundoff
                @test info.error ≈ tail(k) atol=roundoff * norm(expected)
                difference = Matrix(C) - expected
                @test opnorm(difference) ≈ tail(k) atol=roundoff * norm(expected)
                reference = F.U[:, 1:k] * Diagonal(F.S[1:k]) * F.Vt[1:k, :]
                @test Matrix(C) ≈ reference rtol=roundoff atol=roundoff * norm(expected)
                @test (oldU, oldV, sourceU, sourceV) == snapshots
                @test A.U === sourceU
                @test A.V === sourceV
                for old in (oldU, oldV, sourceU, sourceV), new in (C.U, C.V)
                    @test !Base.mightalias(old, new)
                end
                for buffer in ACA._scratcharrays(scratch), factor in (C.U, C.V)
                    @test !Base.mightalias(factor, buffer)
                end
            end
        end

        @testset "$T aliasing and cancellation" begin
            for shared in (:left, :right, :both)
                C = addition_fixture(rng, T, 19, 15, 5)
                A = addition_fixture(rng, T, 19, 15, 5)
                shared in (:left, :both) && (A.U = C.U)
                shared in (:right, :both) && (A.V = C.V)
                factors = (C.U, C.V, A.U, A.V)
                snapshots = copy.(factors)
                expected = Matrix(C) + alpha * Matrix(A)
                scratch = ACA.BlockScratch{T}()
                info = ACA._addlowrank!(C, A, alpha, scratch; rtol=1e-4)
                @test norm(Matrix(C) - expected) <=
                    sqrt(min(size(C)...)) * info.error + roundoff * norm(expected)
                @test factors == snapshots
                @test info.converged
                @test A.U === factors[3]
                @test A.V === factors[4]
            end
            # Separate objects with exactly cancelling shared factors take the
            # general path. Absolute tolerance accounts for QR/LQ roundoff.
            for (m, n, r) in ((19, 15, 5), (3, 4, 5))
                C = addition_fixture(rng, T, m, n, r)
                A = ACA.LowRankMatrix{T}(C.U, C.V)
                scale = norm(Matrix(C))
                info = ACA._addlowrank!(
                    C, A, -one(T), ACA.BlockScratch{T}(); atol=roundoff * scale, rtol=0
                )
                @test rank(C) == 0
                @test info.converged
                @test norm(Matrix(C)) == 0
            end
            # Near cancellation: tolerance is measured against the small sum,
            # while the test's floating-point allowance scales with operands.
            C = addition_fixture(rng, T, 19, 15, 5)
            A = ACA.LowRankMatrix{T}(C.U * T(1 - 0.01), C.V)
            expected = Matrix(C) - Matrix(A)
            allowance = roundoff * (norm(Matrix(C)) + norm(Matrix(A)))
            info = ACA._addlowrank!(C, A, -one(T), ACA.BlockScratch{T}(); rtol=0.01)
            @test norm(Matrix(C) - expected) <= sqrt(15) * info.error + allowance
        end

        @testset "$T zero, empty, and self updates" begin
            # Hierarchical addition scales C += alpha*C exactly instead.
            C = addition_fixture(rng, T, 4, 5, 6)
            @test_throws ArgumentError ACA._addlowrank!(C, C, alpha, ACA.BlockScratch{T}())
            for (m, n, rc, ra) in (
                (12, 9, 0, 4),
                (12, 9, 4, 0),
                (12, 9, 0, 0),
                (0, 9, 4, 3),
                (12, 0, 4, 3),
                (0, 0, 0, 0),
            )
                C = addition_fixture(rng, T, m, n, rc)
                A = addition_fixture(rng, T, m, n, ra)
                expected = Matrix(C) + alpha * Matrix(A)
                info = ACA._addlowrank!(C, A, alpha, ACA.BlockScratch{T}())
                @test Matrix(C) ≈ expected atol=roundoff * max(1, norm(expected))
                @test info.converged
            end
            for coefficient in (zero(T), -one(T)), cap in (0, 2)
                C = addition_fixture(rng, T, 12, 9, 4)
                A = addition_fixture(rng, T, 12, 9, 4)
                old = (C.U, C.V)
                before = copy.(old)
                info = ACA._addlowrank!(
                    C, A, coefficient, ACA.BlockScratch{T}(); rtol=0, maxrank=cap
                )
                @test rank(C) <= cap
                @test old == before
                @test info.rank_limited
                @test !info.converged
            end
        end
    end

    @testset "Validation before mutation" begin
        C = addition_fixture(rng, Float64, 15, 13, 4)
        A = addition_fixture(rng, Float64, 15, 13, 3)
        U, V = C.U, C.V
        scratch = ACA.BlockScratch{Float64}()
        for options in ((; atol=-1), (; rtol=NaN), (; maxrank=-1))
            @test_throws ArgumentError ACA._addlowrank!(C, A, 1, scratch; options...)
        end
        @test_throws InexactError ACA._addlowrank!(C, A, 1im, scratch)
        @test_throws ArgumentError ACA._addlowrank!(C, A, Inf, scratch)
        @test_throws ArgumentError ACA._addlowrank!(C, A, NaN, scratch)
        B = addition_fixture(rng, Float64, 14, 13, 3)
        @test_throws DimensionMismatch ACA._addlowrank!(C, B, 1, scratch)
        A.V = zeros(5, 13)
        @test_throws DimensionMismatch ACA._addlowrank!(C, A, 1, scratch)
        A.V = zeros(3, 13)
        C.V = zeros(2, 13)
        @test_throws DimensionMismatch ACA._addlowrank!(C, A, 1, scratch)
        C.V = V
        @test C.U === U
        @test C.V === V
        @test all(isempty, ACA._scratcharrays(scratch))
    end

    @testset "Capacity reuse, dense shortcut, and allocations" begin
        for T in (Float64, ComplexF64)
            scratch = ACA.BlockScratch{T}()
            C = addition_fixture(rng, T, 64, 48, 8)
            A = addition_fixture(rng, T, 64, 48, 7)
            ACA._addlowrank!(C, A, 1, scratch)
            buffers = ACA._scratcharrays(scratch)
            capacities, pointers = length.(buffers[1:5]), pointer.(buffers[1:5])
            for (m, n, rc, ra) in
                ((23, 19, 3, 4), (23, 4, 3, 4), (4, 23, 3, 4), (4, 5, 3, 4))
                C = addition_fixture(rng, T, m, n, rc)
                A = addition_fixture(rng, T, m, n, ra)
                ACA._addlowrank!(C, A, 1, scratch)
                @test length.(buffers[1:5]) == capacities
                @test pointer.(buffers[1:5]) == pointers
            end
            fresh = ACA.BlockScratch{T}()
            C = addition_fixture(rng, T, 5, 6, 4)
            A = addition_fixture(rng, T, 5, 6, 4)
            ACA._addlowrank!(C, A, 1, fresh)
            @test isempty(fresh.left)
            @test isempty(fresh.right)
            @test length(fresh.small) == 30
            # A zero contribution uses ordinary recompression; in particular it
            # needs no packing buffer on an unfactorized short side.
            fresh = ACA.BlockScratch{T}()
            C = addition_fixture(rng, T, 23, 4, 7)
            A = addition_fixture(rng, T, 23, 4, 5)
            ACA._addlowrank!(C, A, 0, fresh)
            @test length(fresh.left) == 23 * 7
            @test isempty(fresh.right)
            C = addition_fixture(rng, T, 1024, 768, 4)
            A = addition_fixture(rng, T, 1024, 768, 4)
            bytes = addition_allocations(C, A, scratch)
            # Owned outputs and LAPACK work arrays (block size at most 64); no
            # dense m × n temporary.
            @test bytes <= sizeof(T) * (1024 + 768) * (rank(C) + 64)
        end
    end
end
