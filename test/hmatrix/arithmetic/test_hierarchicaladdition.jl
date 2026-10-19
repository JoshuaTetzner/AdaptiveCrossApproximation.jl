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

@testset "hierarchicaladdition.jl" begin
    @testset "_addlowrank!" begin
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
                    for f in fieldnames(ACA.BlockScratch), factor in (C.U, C.V)
                        @test !Base.mightalias(factor, getfield(scratch, f))
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

            @testset "$T zero and empty updates" begin
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
            @test_throws InexactError ACA._addlowrank!(C, A, 1im, scratch)
            B = addition_fixture(rng, Float64, 14, 13, 3)
            @test_throws DimensionMismatch ACA._addlowrank!(C, B, 1, scratch)
            @test C.U === U
            @test C.V === V
            @test all(f -> isempty(getfield(scratch, f)), fieldnames(ACA.BlockScratch))
        end

        @testset "Capacity reuse, dense shortcut, and allocations" begin
            for T in (Float64, ComplexF64)
                scratch = ACA.BlockScratch{T}()
                C = addition_fixture(rng, T, 64, 48, 8)
                A = addition_fixture(rng, T, 64, 48, 7)
                ACA._addlowrank!(C, A, 1, scratch)
                buffers = (
                    scratch.left,
                    scratch.right,
                    scratch.small,
                    scratch.tauleft,
                    scratch.tauright,
                )
                capacities, pointers = length.(buffers), pointer.(buffers)
                for (m, n, rc, ra) in
                    ((23, 19, 3, 4), (23, 4, 3, 4), (4, 23, 3, 4), (4, 5, 3, 4))
                    C = addition_fixture(rng, T, m, n, rc)
                    A = addition_fixture(rng, T, m, n, ra)
                    ACA._addlowrank!(C, A, 1, scratch)
                    @test length.(buffers) == capacities
                    @test pointer.(buffers) == pointers
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

    @testset "_hierarchicaladd!" begin
        ACA = AdaptiveCrossApproximation
        rng = MersenneTwister(3287)
        for T in (Float64, ComplexF64, Float32, ComplexF32)
            alpha = T <: Real ? T(-0.7) : T(-0.7 + 0.4im)
            beta = T <: Real ? T(0.3) : T(0.3 - 0.2im)
            tolerance = 1500eps(real(T))
            options = (; atol=0, rtol=10eps(real(T)), maxrank=typemax(Int))
            @testset "_hierarchicaladd! ($T)" begin
                for reversed in (false, true), self in (false, true)
                    C, original = hierarchical_fixture(T)
                    A = self ? C : hierarchical_source(C; reversed)
                    aoriginal = workspace_test_dense(A, A.rowranges, A.colranges)
                    nearbefore = copy.(A.near)
                    farbefore = [(copy(R.U), copy(R.V)) for R in A.far]
                    info = @inferred hierarchical_add!(
                        C, A, alpha, ACA.BlockScratch{T}(), options
                    )
                    @test info.converged && !info.rank_limited
                    actual = workspace_test_dense(C, C.rowranges, C.colranges)
                    @test actual ≈ original + alpha * aoriginal rtol=tolerance
                    if !self
                        @test A.near == nearbefore
                        @test [(R.U, R.V) for R in A.far] == farbefore
                    end
                end
                C, original = hierarchical_fixture(T)
                A = hierarchical_source(C)
                capped = (; atol=0, rtol=0, maxrank=0)
                info = hierarchical_add!(C, A, alpha, ACA.BlockScratch{T}(), capped)
                @test info.rank_limited && !info.converged
                @test all(iszero ∘ rank, C.far)
                @test info.max_block_error > 0
                before = copy.(C.near)
                info = hierarchical_add!(C, A, 0, ACA.BlockScratch{T}(), capped)
                @test info.converged
                @test C.near == before
            end
        end

        @testset "_checkadditionpartition / _hierarchicaladd! (validation)" begin
            C, D = hierarchical_fixture(ComplexF64)
            options = (; atol=0, rtol=1e-12, maxrank=typemax(Int))
            nearbefore = copy.(C.near)
            @test_throws ArgumentError hierarchical_add!(
                C, C, Inf, ACA.BlockScratch{ComplexF64}(), options
            )
            changed = hierarchical_source(C)
            changed.csr.isfar[1] = !changed.csr.isfar[1]
            @test_throws ArgumentError hierarchical_add!(
                C, changed, 1, ACA.BlockScratch{ComplexF64}(), options
            )
            @test C.near == nearbefore
        end
    end
end
