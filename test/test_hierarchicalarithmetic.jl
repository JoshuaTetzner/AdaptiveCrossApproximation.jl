using AdaptiveCrossApproximation
using H2Trees
using LinearAlgebra
using Random
using Test

# Reuse the exact, irregular-tree fixture from test_treeview.jl with full storage.
function hierarchical_core(T; ordering=AdaptiveCrossApproximation.PreserveSpaceOrder())
    fixture = arithmetic_workspace_fixture(ordering)
    core = fixture.core
    convertmatrix(A) = T <: Real ? Matrix{T}(real.(A)) : Matrix{T}(A)
    near = map(convertmatrix, core.near)
    far = [
        AdaptiveCrossApproximation.LowRankMatrix{T}(convertmatrix(R.U), convertmatrix(R.V))
        for R in core.far
    ]
    return AdaptiveCrossApproximation.HMatrixCore{T}(
        core.tree, near, far, core.csr, core.csc, core.ordering, size(core), core.tol
    )
end

# Tree-order view with its dense tree-order reference, built from the blocks.
function hierarchical_fixture(T; kwargs...)
    W = AdaptiveCrossApproximation._TreeView(hierarchical_core(T; kwargs...))
    D = workspace_test_dense(W, W.rowranges, W.colranges)
    return W, D
end

# The same logical matrix on the same tree, optionally with reversed physical
# storage order.
function hierarchical_source(W; reversed=true)
    ACA = AdaptiveCrossApproximation
    csr = W.csr
    near = [copy(D) for D in W.near]
    far = [ACA.LowRankMatrix(copy(R.U), copy(R.V)) for R in W.far]
    indices = copy(csr.storageidx)
    if reversed
        reverse!(near)
        reverse!(far)
        for k in eachindex(indices)
            indices[k] = (csr.isfar[k] ? length(far) : length(near)) + 1 - indices[k]
        end
    end
    index = ACA.BlockCSR(
        copy(csr.rowptr), copy(csr.colnodes), copy(csr.isfar), copy(csr.transposed), indices
    )
    core = ACA.HMatrixCore{eltype(W)}(
        W.tree, near, far, index, W.csc, ACA.PreserveSpaceOrder(), size(W), 1e-10
    )
    return ACA._TreeView(core)
end

function hierarchical_thin_bytes(Y, H, X, scratch, side)
    if side === :left
        AdaptiveCrossApproximation._thinleft!(Y, H, X, 0.3, 0, scratch)
        return @allocated AdaptiveCrossApproximation._thinleft!(Y, H, X, 0.3, 0, scratch)
    end
    AdaptiveCrossApproximation._thinright!(Y, X, H, 0.3, 0, scratch)
    return @allocated AdaptiveCrossApproximation._thinright!(Y, X, H, 0.3, 0, scratch)
end

@testset "Hierarchical arithmetic kernels" begin
    ACA = AdaptiveCrossApproximation
    rng = MersenneTwister(3287)
    for T in (Float64, ComplexF64, Float32, ComplexF32)
        alpha = T <: Real ? T(-0.7) : T(-0.7 + 0.4im)
        beta = T <: Real ? T(0.3) : T(0.3 - 0.2im)
        tolerance = 1500eps(real(T))
        options = (; atol=0, rtol=10eps(real(T)), maxrank=typemax(Int))
        @testset "$T addition" begin
            for reversed in (false, true), self in (false, true)
                C, original = hierarchical_fixture(T)
                A = self ? C : hierarchical_source(C; reversed)
                aoriginal = workspace_test_dense(A, A.rowranges, A.colranges)
                nearbefore = copy.(A.near)
                farbefore = [(copy(R.U), copy(R.V)) for R in A.far]
                info = @inferred ACA._hierarchicaladd!(
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
            info = ACA._hierarchicaladd!(C, A, alpha, ACA.BlockScratch{T}(), capped)
            @test info.rank_limited && !info.converged
            @test all(iszero ∘ rank, C.far)
            @test info.max_block_error > 0
            before = copy.(C.near)
            info = ACA._hierarchicaladd!(C, A, 0, ACA.BlockScratch{T}(), capped)
            @test info.converged
            @test C.near == before
        end

        @testset "$T thin panels" begin
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
                            !isempty(ACA._overlap(W.colranges[W.csr.colnodes[k]], H.cols))
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
                    ACA._thinexecute!(Y, X, H, alpha, scratch.small, Val(:T), Val(:left))
                    @test Y ≈ alpha * transpose(ref) * X rtol=tolerance atol=tolerance
                end
                @test W.near == nearbefore
                @test [(R.U, R.V) for R in W.far] == farbefore
            end
        end
    end

    @testset "Validation and warmed storage" begin
        C, D = hierarchical_fixture(ComplexF64)
        options = (; atol=0, rtol=1e-12, maxrank=typemax(Int))
        nearbefore = copy.(C.near)
        A = hierarchical_source(C; reversed=false)
        A.near[1] = C.near[1]
        @test_throws ArgumentError ACA._hierarchicaladd!(
            C, A, 1, ACA.BlockScratch{ComplexF64}(), options
        )
        @test C.near == nearbefore
        @test_throws ArgumentError ACA._hierarchicaladd!(
            C, C, Inf, ACA.BlockScratch{ComplexF64}(), options
        )
        changed = hierarchical_source(C)
        changed.csr.isfar[1] = !changed.csr.isfar[1]
        @test_throws ArgumentError ACA._hierarchicaladd!(
            C, changed, 1, ACA.BlockScratch{ComplexF64}(), options
        )
        @test C.near == nearbefore
        @test_throws BoundsError ACA.blockview(C, 0, 1)
        @test_throws BoundsError ACA.blockview(C, 0:3, 1:2)
        H = ACA.blockview(C)
        X = randn(rng, ComplexF64, size(C, 2), 3)
        Y = zeros(ComplexF64, size(C, 1), 3)
        scratch = ACA.BlockScratch{ComplexF64}()
        @test_throws DimensionMismatch ACA._thinleft!(
            Y, H, X[1:(end - 1), :], 1, 0, scratch
        )
        @test_throws ArgumentError ACA._thinleft!(Y, H, X, Inf, 0, scratch)
        @test all(iszero, Y) && isempty(scratch.small)
        k = findfirst(!, C.csr.isfar)
        aliasY = C.near[C.csr.storageidx[k]]
        sub = ACA.blockview(C, C.entryrows[k], C.csr.colnodes[k])
        @test_throws ArgumentError ACA._thinleft!(
            aliasY,
            sub,
            randn(rng, ComplexF64, size(sub, 2), size(aliasY, 2)),
            1,
            0,
            scratch,
        )
        for T in (Float64, ComplexF64)
            W, dense = hierarchical_fixture(T)
            for H in (ACA.blockview(W), ACA.blockview(W, 2:5, 2:4)), side in (:left, :right)
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
            scratch = ACA.BlockScratch{T}()
            resize!(scratch.left, length(X))
            borrowed = reshape(view(scratch.left, 1:length(X)), size(X))
            snapshot = copy(Y)
            @test_throws ArgumentError ACA._thinleft!(Y, H, borrowed, 1, 0, scratch)
            @test Y == snapshot
        end
    end
end
