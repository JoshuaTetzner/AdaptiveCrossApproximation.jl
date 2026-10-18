using AdaptiveCrossApproximation
using H2Trees
using LinearAlgebra
using Random
using Test

# Independently assembled exact blocks; no ACA or new product kernel is used
# for reference construction. Axes/frontiers and dense/far classifications vary.
function product_test_fixture(
    rng, T, rowtree, coltree; rowfront=:children, colfront=:leaves, kind=:mixed
)
    ACA = AdaptiveCrossApproximation
    frontier(tree, choice) =
        if choice === :root
            [H2Trees.root(tree)]
        elseif choice === :leaves
            collect(H2Trees.leaves(tree))
        else
            collect(H2Trees.children(tree, H2Trees.root(tree)))
        end
    rr, cc = ACA._clusterranges(rowtree), ACA._clusterranges(coltree)
    m, n = length(rr[H2Trees.root(rowtree)]), length(cc[H2Trees.root(coltree)])
    left, right = randn(rng, T, m, 2), randn(rng, T, 2, n)
    near, far = Matrix{T}[], ACA.LowRankMatrix{T}[]
    rows, cols, indices, flags = Int[], Int[], Int[], BitVector()
    reference = zeros(T, m, n)
    for (i, row) in enumerate(frontier(rowtree, rowfront)),
        (j, col) in enumerate(frontier(coltree, colfront))

        isfar = kind === :far || kind === :mixed && isodd(i + j)
        U, V = left[rr[row], :], right[:, cc[col]]
        if isfar
            reference[rr[row], cc[col]] = U * V
            push!(far, ACA.LowRankMatrix(U, V))
        else
            D = U * V + randn(rng, T, length(rr[row]), length(cc[col]))
            reference[rr[row], cc[col]] = D
            push!(near, D)
        end
        push!(rows, row)
        push!(cols, col)
        push!(indices, isfar ? length(far) : length(near))
        push!(flags, isfar)
    end
    csr, csc = ACA._blockindices(
        rows,
        cols,
        flags,
        falses(length(flags)),
        indices,
        H2Trees.numberofnodes(rowtree),
        H2Trees.numberofnodes(coltree),
    )
    core = ACA.HMatrixCore{T}(
        H2Trees.BlockTree(rowtree, coltree),
        near,
        far,
        csr,
        csc,
        ACA.PreserveSpaceOrder(),
        (m, n),
        1e-10,
    )
    return ACA._TreeView(core), reference
end

product_test_dense(W) = workspace_test_dense(W, W.rowranges, W.colranges)

product_test_block(H::AdaptiveCrossApproximation._TreeView) =
    AdaptiveCrossApproximation.blockview(H)
product_test_block(H) = H

function product_test_plan(C, A, B; kwargs...)
    return AdaptiveCrossApproximation._ProductPlan(
        product_test_block(C), product_test_block(A), product_test_block(B); kwargs...
    )
end

function product_test_addmul!(P, alpha, scratch; atol=0, rtol=1e-6, maxrank=typemax(Int))
    return AdaptiveCrossApproximation._addmul!(P, alpha, scratch, (; atol, rtol, maxrank))
end

function product_test_addmul!(C, A, B, alpha, scratch; batchrank=128, kwargs...)
    return product_test_addmul!(
        product_test_plan(C, A, B; batchrank), alpha, scratch; kwargs...
    )
end

product_test_arrays(scratch) =
    (scratch.productleft, scratch.productright, scratch.aggregate)

function product_test_execution_bytes(P, scratch, status, options)
    ACA = AdaptiveCrossApproximation
    check, execute = ACA._checkproductworkspace, ACA._executeproduct!
    alpha = eltype(scratch.left)(0.2)
    check(P, scratch)
    execute(P, alpha, scratch, status, options)
    validation = @allocated check(P, scratch)
    execution = @allocated execute(P, alpha, scratch, status, options)
    return validation, execution
end

@testset "Hierarchical multiplication" begin
    ACA = AdaptiveCrossApproximation
    rng = MersenneTwister(74828)
    seed = arithmetic_workspace_fixture(ACA.PreserveSpaceOrder())
    rowtree, inner = H2Trees.testtree(seed.core.tree), H2Trees.trialtree(seed.core.tree)
    for T in (Float64, ComplexF64, Float32, ComplexF32)
        tolerance = 2000eps(real(T))
        alpha = T <: Real ? T(-0.3) : T(-0.3 + 0.2im)
        @testset "$T differing partitions" begin
            for (afront, bfront) in (
                ((:children, :leaves), (:leaves, :children)),
                ((:leaves, :children), (:children, :leaves)),
                ((:root, :leaves), (:children, :root)),
            )
                A, da = product_test_fixture(
                    rng, T, rowtree, inner; rowfront=afront[1], colfront=afront[2]
                )
                B, db = product_test_fixture(
                    rng, T, inner, rowtree; rowfront=bfront[1], colfront=bfront[2]
                )
                C, dc = product_test_fixture(
                    rng, T, rowtree, rowtree; rowfront=:children, colfront=:children
                )
                acopy, bcopy = product_test_dense(A), product_test_dense(B)
                P = product_test_plan(C, A, B; batchrank=7)
                scratch = ACA.BlockScratch{T}()
                info = product_test_addmul!(P, alpha, scratch; rtol=10eps(real(T)))
                @test info.converged && !info.rank_limited
                expected = dc + alpha * da * db
                @test product_test_dense(C) ≈ expected rtol=tolerance
                @test product_test_dense(A) == acopy
                @test product_test_dense(B) == bcopy
                pointers = pointer.(product_test_arrays(scratch))
                product_test_addmul!(P, alpha, scratch; rtol=10eps(real(T)))
                @test product_test_dense(C) ≈ expected + alpha * da * db rtol=tolerance
                @test pointer.(product_test_arrays(scratch)) == pointers
                # Ranks are deliberately changed after structural planning.
                for R in A.far
                    ACA._recompress!(R, scratch; maxrank=0)
                end
                before = product_test_dense(C)
                changed = product_test_dense(A)
                product_test_addmul!(P, alpha, scratch; rtol=10eps(real(T)))
                @test product_test_dense(C) ≈ before + alpha * changed * db rtol=tolerance
            end
        end
        @testset "$T restricted views and dense/dense to far" begin
            A, da = product_test_fixture(
                rng, T, rowtree, inner; kind=:dense, rowfront=:root
            )
            B, db = product_test_fixture(
                rng, T, inner, rowtree; kind=:dense, colfront=:root
            )
            for batchrank in (1, 5, 32)
                C, dc = product_test_fixture(
                    rng, T, rowtree, rowtree; kind=:far, rowfront=:root, colfront=:root
                )
                cr, innerpart, cc = 3:12, 2:9, 2:14
                info = product_test_addmul!(
                    ACA.blockview(C, cr, cc),
                    ACA.blockview(A, cr, innerpart),
                    ACA.blockview(B, innerpart, cc),
                    alpha,
                    ACA.BlockScratch{T}();
                    batchrank,
                    rtol=10eps(real(T)),
                )
                expected = copy(dc)
                expected[cr, cc] += alpha * da[cr, innerpart] * db[innerpart, cc]
                @test product_test_dense(C) ≈ expected rtol=tolerance
                @test info.flushes == cld(length(innerpart), batchrank)
            end
            C, dc = product_test_fixture(rng, T, rowtree, rowtree; kind=:far)
            info = product_test_addmul!(
                C, A, B, alpha, ACA.BlockScratch{T}(); maxrank=0, rtol=0
            )
            @test info.rank_limited && !info.converged
            @test all(iszero ∘ rank, C.far)
            # Empty contracted intervals do not touch even the initial far ranks.
            C, dc = product_test_fixture(rng, T, rowtree, rowtree; kind=:far)
            info = product_test_addmul!(
                ACA.blockview(C),
                ACA.blockview(A, 1:16, 2:1),
                ACA.blockview(B, 2:1, 1:16),
                alpha,
                ACA.BlockScratch{T}(),
            )
            @test info.flushes == 0
            @test product_test_dense(C) == dc
        end
        @testset "$T adaptive dense core and reusable scratch" begin
            A, da = product_test_fixture(
                rng, T, rowtree, inner; kind=:far, rowfront=:root, colfront=:root
            )
            B, db = product_test_fixture(
                rng, T, inner, inner; kind=:far, rowfront=:root, colfront=:root
            )
            C, dc = product_test_fixture(
                rng, T, rowtree, inner; kind=:far, rowfront=:root, colfront=:root
            )
            P = product_test_plan(C, A, B; batchrank=128)
            shared = ACA.BlockScratch{T}()
            @test product_test_addmul!(P, alpha, shared; rtol=10eps(real(T))).flushes == 1
            @test P.groups[1].capacity == 2
            @test (length(shared.productleft), length(shared.productright)) == (32, 24)
            @test product_test_dense(C) ≈ dc + alpha * da * db rtol=tolerance
            # Oversized factors must not invalidate the plan or rank policy.
            A.far[1].U = randn(rng, T, 16, 20)
            A.far[1].V = randn(rng, T, 20, 12)
            changed = product_test_dense(A)
            before = product_test_dense(C)
            product_test_addmul!(P, alpha, ACA.BlockScratch{T}(); rtol=10eps(real(T)))
            @test P.groups[1].capacity == 2
            @test product_test_dense(C) ≈ before + alpha * changed * db rtol=tolerance
            B.far[1].U = randn(rng, T, 12, 15)
            B.far[1].V = randn(rng, T, 15, 12)
            changedb = product_test_dense(B)
            before = product_test_dense(C)
            product_test_addmul!(P, alpha, ACA.BlockScratch{T}(); rtol=10eps(real(T)))
            @test P.groups[1].capacity == 15
            @test product_test_dense(C) ≈ before + alpha * changed * changedb rtol=tolerance
            other, originalother = product_test_fixture(rng, T, rowtree, inner)
            # Plans share the buffers of the scratch they execute with.
            Q = product_test_plan(other, A, B)
            product_test_addmul!(P, alpha, shared; rtol=10eps(real(T)))
            pointers = pointer.(product_test_arrays(shared))
            originalC = product_test_dense(C)
            originalother = product_test_dense(other)
            product_test_addmul!(Q, alpha, shared; rtol=10eps(real(T)))
            @test product_test_dense(other) ≈ originalother + alpha * changed * changedb rtol=tolerance
            @test product_test_dense(C) == originalC
            @test pointer.(product_test_arrays(shared)) == pointers
            # Dense destinations perform no recompression; validation and the
            # execution loop must both reuse storage after warming.
            C, dc = product_test_fixture(rng, T, rowtree, inner; kind=:dense)
            P = product_test_plan(C, A, B)
            scratch = ACA.BlockScratch{T}()
            status = ACA._ProductStatus(true, false, zero(real(T)), 0)
            options = (; atol=0, rtol=1e-6, maxrank=typemax(Int))
            product_test_execution_bytes(P, scratch, status, options)
            @test product_test_execution_bytes(P, scratch, status, options) == (0, 0)
        end
    end
    @testset "Validation before writes" begin
        T = ComplexF64
        A, _ = product_test_fixture(rng, T, rowtree, rowtree)
        B, _ = product_test_fixture(rng, T, rowtree, rowtree)
        C, original = product_test_fixture(rng, T, rowtree, rowtree)
        scratch = ACA.BlockScratch{T}()
        P = product_test_plan(C, A, B)
        @test_throws ArgumentError product_test_addmul!(P, Inf, scratch)
        @test_throws ArgumentError product_test_plan(C, A, B; batchrank=0)
        @test_throws DimensionMismatch product_test_plan(ACA.blockview(C, 1:3, 1:16), A, B)
        info = product_test_addmul!(P, 0, scratch)
        @test info.flushes == 0
        @test product_test_dense(C) == original
        @test_throws ArgumentError product_test_addmul!(A, A, B, 1, scratch)
        aliased = hierarchical_source(C; reversed=false)
        aliased.near[1] = A.near[1]
        @test_throws ArgumentError product_test_addmul!(aliased, A, B, 1, scratch)
        scratch.left = vec(A.near[1])
        @test_throws ArgumentError product_test_addmul!(P, 1, scratch)
        @test product_test_dense(C) == original
        # Equivalent geometries with unrelated tree objects are not compatible.
        unrelated, _ = hierarchical_fixture(T)
        @test_throws ArgumentError product_test_plan(C, A, unrelated)
        Z, _ = product_test_fixture(rng, T, rowtree, rowtree; kind=:far)
        target, _ = product_test_fixture(rng, T, rowtree, rowtree; kind=:far)
        ACA._recompress!(Z.far[1], ACA.BlockScratch{T}(); maxrank=0)
        target.far[1] = Z.far[1]
        @test_throws ArgumentError product_test_addmul!(
            target, Z, Z, 1, ACA.BlockScratch{T}()
        )
    end
    @testset "In-place Schur update on disjoint workspace blocks" begin
        W, original = product_test_fixture(
            rng, ComplexF64, rowtree, rowtree; rowfront=:children, colfront=:children
        )
        children = collect(H2Trees.children(rowtree, H2Trees.root(rowtree)))
        left, right = W.rowranges[children[1]], W.rowranges[children[2]]
        expected = copy(original)
        expected[right, right] -= original[right, left] * original[left, right]
        info = product_test_addmul!(
            ACA.blockview(W, right, right),
            ACA.blockview(W, right, left),
            ACA.blockview(W, left, right),
            -1,
            ACA.BlockScratch{ComplexF64}();
            rtol=1e-14,
        )
        @test info.converged
        @test product_test_dense(W) ≈ expected rtol=1e-12
    end
end
