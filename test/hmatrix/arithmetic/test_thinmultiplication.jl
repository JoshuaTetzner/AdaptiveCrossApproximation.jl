using AdaptiveCrossApproximation
using H2Trees
using LinearAlgebra
using Random
using Test

@testset "thinmultiplication.jl" begin
    @testset "_thinleft! / _thinright!" begin
        ACA = AdaptiveCrossApproximation
        rng = MersenneTwister(3287)
        for T in (Float64, ComplexF64, Float32, ComplexF32)
            alpha = T <: Real ? T(-0.7) : T(-0.7 + 0.4im)
            beta = T <: Real ? T(0.3) : T(0.3 - 0.2im)
            tolerance = 1500eps(real(T))
            options = (; atol=0, rtol=10eps(real(T)), maxrank=typemax(Int))
            @testset "_thinleft! / _thinright! ($T)" begin
                for ordering in (ACA.PreserveSpaceOrder(), ACA.PermuteSpaceInPlace())
                    W, D = hierarchical_fixture(T; ordering)
                    m, n = size(W)
                    rowtree, coltree = H2Trees.testtree(W.tree), H2Trees.trialtree(W.tree)
                    @test W.rowranges[H2Trees.root(rowtree)] == 1:m
                    @test W.colranges[H2Trees.root(coltree)] == 1:n
                    rowleaf = first(H2Trees.leaves(rowtree))
                    colleaf = last(collect(H2Trees.leaves(coltree)))
                    views = (
                        ACA.blockview(W),
                        ACA.blockview(W, rowleaf, colleaf),
                        ACA.blockview(W, 2:(m - 1), 2:(n - 1)),
                        ACA.blockview(W, 1:m, 2:3),
                        ACA.blockview(W, 2:1, 1:n),
                        ACA.blockview(W, 1:m, 2:1),
                    )
                    nearbefore, farbefore = copy.(W.near),
                    [(copy(R.U), copy(R.V)) for R in W.far]
                    scratch = ACA.BlockScratch{T}()
                    for H in views
                        expectedentries = [
                            k for k in 1:length(W.csr) if
                            !isempty(ACA._overlap(W.rowranges[W.entryrows[k]], H.rows)) &&
                                !isempty(
                                    ACA._overlap(W.colranges[W.csr.colnodes[k]], H.cols)
                                )
                        ]
                        @test collect(H.entries) == expectedentries
                        ref = D[H.rows, H.cols]
                        for side in (:left, :right), q in (0, 1, 4), padded in (false, true)
                            inputdims = side === :left ? (size(H, 2), q) : (q, size(H, 1))
                            outputdims = side === :left ? (size(H, 1), q) : (q, size(H, 2))
                            P = randn(rng, T, inputdims[1] + 2, inputdims[2] + 2)
                            X = if padded
                                view(P, 2:(inputdims[1] + 1), 2:(inputdims[2] + 1))
                            else
                                randn(rng, T, inputdims...)
                            end
                            Q = randn(rng, T, outputdims[1] + 2, outputdims[2] + 2)
                            Y = if padded
                                view(Q, 2:(outputdims[1] + 1), 2:(outputdims[2] + 1))
                            else
                                randn(rng, T, outputdims...)
                            end
                            xsnapshot = copy(X)
                            if side === :left
                                expected = alpha * ref * X + beta * Y
                                @test ACA._thinleft!(Y, H, X, alpha, beta, scratch) === Y
                            else
                                expected = alpha * X * ref + beta * Y
                                @test ACA._thinright!(Y, X, H, alpha, beta, scratch) === Y
                            end
                            @test Y ≈ expected rtol=tolerance atol=tolerance * max(
                                norm(expected), one(real(T))
                            )
                            @test X == xsnapshot
                        end
                        # Transposed application, as used by transposed triangular solves.
                        X = randn(rng, T, size(H, 1), 3)
                        Y = zeros(T, size(H, 2), 3)
                        ACA._growscratch!(scratch.small, 3 * maximum(rank, W.far; init=0))
                        ACA._thinexecute!(
                            Y, X, H, alpha, scratch.small, Val(:T), Val(:left)
                        )
                        @test Y ≈ alpha * transpose(ref) * X rtol=tolerance atol=tolerance
                    end
                    @test W.near == nearbefore
                    @test [(R.U, R.V) for R in W.far] == farbefore
                end
            end
        end

        @testset "_thinleft! / _thinright! (validation and capacity reuse)" begin
            C, D = hierarchical_fixture(ComplexF64)
            options = (; atol=0, rtol=1e-12, maxrank=typemax(Int))
            nearbefore = copy.(C.near)
            H = ACA.blockview(C)
            X = randn(rng, ComplexF64, size(C, 2), 3)
            Y = zeros(ComplexF64, size(C, 1), 3)
            scratch = ACA.BlockScratch{ComplexF64}()
            @test_throws DimensionMismatch ACA._thinleft!(
                Y, H, X[1:(end - 1), :], 1, 0, scratch
            )
            @test all(iszero, Y) && isempty(scratch.small)
            for T in (Float64, ComplexF64)
                W, dense = hierarchical_fixture(T)
                for H in (ACA.blockview(W), ACA.blockview(W, 2:5, 2:4)),
                    side in (:left, :right)

                    q = 3
                    inputdims = side === :left ? (size(H, 2), q) : (q, size(H, 1))
                    outputdims = side === :left ? (size(H, 1), q) : (q, size(H, 2))
                    X, Y = randn(rng, T, inputdims...), zeros(T, outputdims...)
                    scratch = ACA.BlockScratch{T}()
                    bytes = hierarchical_thin_bytes(Y, H, X, scratch, side)
                    @test bytes == 0
                    pointers = pointer.((scratch.left, scratch.small))
                    hierarchical_thin_bytes(Y, H, X, scratch, side)
                    @test pointer.((scratch.left, scratch.small)) == pointers
                end
                # A block view holds structural references, not rank-dependent views.
                H = ACA.blockview(W, 2:(size(W, 1) - 1), 2:(size(W, 2) - 1))
                entries = H.entries
                ACA._recompress!(W.far[1], ACA.BlockScratch{T}(); maxrank=0)
                updated = workspace_test_dense(W, W.rowranges, W.colranges)[H.rows, H.cols]
                X, Y = randn(rng, T, size(H, 2), 3), zeros(T, size(H, 1), 3)
                ACA._thinleft!(Y, H, X, 1, 0, ACA.BlockScratch{T}())
                @test Y ≈ updated * X rtol=1e-11
                @test H.entries === entries
            end
        end
    end
end
