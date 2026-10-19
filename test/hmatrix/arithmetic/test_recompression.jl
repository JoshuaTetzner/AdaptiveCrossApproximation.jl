using AdaptiveCrossApproximation
using LinearAlgebra
using Random
using Test

# Prescribed singular values, arbitrary complex phases, and redundant factor
# directions. All dense products here are test references, not kernel storage.
function recompression_fixture(rng, T, m, n, r; deficient=false)
    d = min(m, n, r)
    Qleft = Matrix(qr(randn(rng, T, m, d)).Q)[:, 1:d]
    Qright = Matrix(qr(randn(rng, T, n, d)).Q)[:, 1:d]
    s = [0.3^(j - 1) for j in 1:d]
    deficient && (s[(d ÷ 2 + 1):end] .= 0)
    U, V = zeros(T, m, r), zeros(T, r, n)
    U[:, 1:d] = Qleft * Diagonal(sqrt.(s))
    V[1:d, :] = Diagonal(sqrt.(s)) * adjoint(Qright)
    W = Matrix(qr(randn(rng, T, r, r)).Q)
    return AdaptiveCrossApproximation.LowRankMatrix(U * W, adjoint(W) * V)
end

function recompression_dense_rank(s, atol, rtol, maxrank)
    tol = max(atol, rtol * first(s))
    tail(k) = k == length(s) ? zero(eltype(s)) : s[k + 1]
    required = findfirst(k -> tail(k) <= tol, 0:length(s)) - 1
    k = min(required, maxrank)
    return k, tail(k), tol, k < required
end

function recompression_allocations(R, scratch)
    U, V = R.U, R.V
    AdaptiveCrossApproximation._recompress!(R, scratch; rtol=0)
    R.U, R.V = U, V
    return @allocated AdaptiveCrossApproximation._recompress!(R, scratch; rtol=0)
end

@testset "recompression.jl" begin
    @testset "_recompress!" begin
        ACA = AdaptiveCrossApproximation
        rng = MersenneTwister(517)
        shapes = (
            (27, 21, 8),
            (27, 5, 8),
            (5, 27, 8),
            (5, 6, 8),
            (8, 19, 8),
            (19, 8, 8),
            (8, 8, 8),
            (1, 9, 3),
            (9, 1, 3),
        )
        for T in (Float64, ComplexF64, Float32, ComplexF32)
            @testset "$T" begin
                scratch = ACA.BlockScratch{T}()
                roundoff = 150eps(real(T))
                for (m, n, r) in shapes, deficient in (false, true)
                    R = recompression_fixture(rng, T, m, n, r; deficient)
                    originalU, originalV = R.U, R.V
                    Ubefore, Vbefore = copy(R.U), copy(R.V)
                    A = Matrix(R)
                    F = svd(A)
                    atol, rtol = 1e-4, 0.035
                    maxrank = deficient ? 2 : typemax(Int)
                    k, err, tol, limited = recompression_dense_rank(
                        F.S, atol, rtol, maxrank
                    )
                    result = @inferred ACA._recompress!(R, scratch; atol, rtol, maxrank)
                    @test rank(R) == result.rank == k
                    @test size(R.U) == (m, k)
                    @test size(R.V) == (k, n)
                    @test result.rank_limited == limited
                    @test result.converged == !limited
                    @test result.error ≈ err atol=roundoff
                    @test result.tolerance ≈ tol atol=roundoff
                    difference = Matrix(R) - A
                    @test opnorm(difference) ≈ err atol=roundoff
                    reference = F.U[:, 1:k] * Diagonal(F.S[1:k]) * F.Vt[1:k, :]
                    @test Matrix(R) ≈ reference atol=roundoff
                    @test originalU == Ubefore
                    @test originalV == Vbefore
                    @test R.U !== originalU
                    @test R.V !== originalV
                    for f in fieldnames(ACA.BlockScratch)
                        @test !Base.mightalias(R.U, getfield(scratch, f))
                        @test !Base.mightalias(R.V, getfield(scratch, f))
                    end
                end

                @testset "Capacity reuse and omitted factorizations" begin
                    fresh = ACA.BlockScratch{T}()
                    R = recompression_fixture(rng, T, 40, 32, 9)
                    ACA._recompress!(R, fresh)
                    buffers = Tuple(
                        getfield(fresh, f) for f in fieldnames(ACA.BlockScratch)
                    )
                    capacities = length.(buffers)
                    pointers = pointer.(buffers)
                    for (m, n, r) in ((23, 17, 7), (20, 3, 7), (3, 20, 7), (3, 4, 7))
                        R = recompression_fixture(rng, T, m, n, r)
                        ACA._recompress!(R, fresh)
                        @test length.(buffers) == capacities
                        @test pointer.(buffers) == pointers
                    end
                    for (m, n, r) in shapes
                        fresh = ACA.BlockScratch{T}()
                        R = recompression_fixture(rng, T, m, n, r)
                        ACA._recompress!(R, fresh)
                        @test length(fresh.left) == (r < m ? m * r : 0)
                        @test length(fresh.right) == (r < n ? n * r : 0)
                        @test length(fresh.small) == min(m, r) * min(n, r)
                        @test length(fresh.tauleft) == (r < m ? r : 0)
                        @test length(fresh.tauright) == (r < n ? r : 0)
                    end
                end

                @testset "Zero, empty, and rank caps" begin
                    for (m, n, r) in
                        ((12, 10, 4), (0, 10, 4), (10, 0, 4), (12, 10, 0), (0, 0, 0))
                        R = ACA.LowRankMatrix(zeros(T, m, r), randn(rng, T, r, n))
                        result = ACA._recompress!(R, scratch; atol=0, rtol=0)
                        @test result.rank == 0
                        @test result.error == 0
                        @test result.converged
                        @test !result.rank_limited
                        @test size(R.U) == (m, 0)
                        @test size(R.V) == (0, n)
                        @test Matrix(R) == zeros(T, m, n)
                    end
                    for cap in (0, 1, 4)
                        R = recompression_fixture(rng, T, 15, 13, 6)
                        result = ACA._recompress!(R, scratch; atol=0, rtol=0, maxrank=cap)
                        @test result.rank == cap
                        @test result.rank_limited
                        @test !result.converged
                    end
                    R = recompression_fixture(rng, T, 15, 13, 6)
                    result = ACA._recompress!(R, scratch; atol=2, rtol=0)
                    @test result.rank == 0
                    @test result.converged
                    @test !result.rank_limited
                end
            end
        end

        @testset "Spectral tolerance" begin
            # The individual tail values meet the spectral tolerance, although
            # their combined Frobenius norm would not.
            R = ACA.LowRankMatrix(
                Matrix(Diagonal([1.0, 0.08, 0.08])), Matrix{Float64}(I, 3, 3)
            )
            result = ACA._recompress!(R, ACA.BlockScratch{Float64}(); atol=0.1, rtol=0)
            @test result.rank == 1
            @test result.converged
        end

        @testset "Scaling and allocation bounds" begin
            for T in (Float64, ComplexF64), scale in (1e-200, 1e200)
                U = Matrix{T}(Diagonal(scale .* [1.0, 0.08, 0.08]))
                R = ACA.LowRankMatrix(U, Matrix{T}(I, 3, 3))
                result = ACA._recompress!(R, ACA.BlockScratch{T}(); atol=0.1scale, rtol=0)
                @test result.rank == 1
                @test result.converged
                @test isfinite(result.error)
                @test result.error / scale ≈ 0.08
            end
            # A deliberately loose bound guards against an accidental full-block
            # allocation, without encoding Julia/LAPACK-version-dependent counts.
            for T in (Float64, ComplexF64)
                R = ACA.LowRankMatrix(randn(rng, T, 1024, 8), randn(rng, T, 8, 768))
                scratch = ACA.BlockScratch{T}()
                recompression_allocations(R, scratch)
                @test recompression_allocations(R, scratch) < sizeof(T) * prod(size(R))
            end
        end
    end

    @testset "recompress!" begin
        ACA = AdaptiveCrossApproximation
        for symmetric in (false, true)
            fixture = arithmetic_workspace_fixture(ACA.PreserveSpaceOrder(); symmetric)
            core = fixture.core
            before = [(copy(R.U), copy(R.V)) for R in core.far]
            H = copy(core)
            ranks = map(rank, H.far)
            @test recompress!(H, ACA.RecompressionWorkspace(H)) === H
            @test all(map(rank, H.far) .<= ranks)
            @test Matrix(H) ≈ fixture.reference rtol = 1e-8
            @test all(R.U == U && R.V == V for (R, (U, V)) in zip(core.far, before))
            G = copy(core)
            @test recompress!(G) === G
            @test Matrix(G) ≈ Matrix(H)
        end
    end
end
