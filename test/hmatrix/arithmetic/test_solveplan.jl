using AdaptiveCrossApproximation
using H2Trees
using LinearAlgebra
using Random
using StaticArrays
using Test

# Each free-axis cluster independently selects a solve-axis frontier. Thus one
# RHS terminal may span several elimination subtrees while neighboring columns
# use finer blocks. No solve/product/recompression generates the reference.
function solve_test_fixture(
    rng, T, H; side=:left, pattern=:mixed, kind=:mixed, freefront=:leaves
)
    ACA = AdaptiveCrossApproximation
    cluster = H2Trees.testtree(H.workspace.tree)
    points = [SVector(Float64(i)^2, 0.0) for i in randperm(rng, 11)]
    free = H2Trees.buildtree(points; builder=H2Trees.TwoNTreeBuilder(; minvalues=2))
    root, freeroot = H.node, H2Trees.root(free)
    freeclusters = freefront === :root ? [freeroot] : collect(H2Trees.leaves(free))
    diagonal = [H.workspace.entryrows[H.leafrecords[i]] for i in H.leafspan]
    children = collect(H2Trees.children(cluster, root))
    isempty(children) && (children = [root])
    near, far = Matrix{T}[], ACA.LowRankMatrix{T}[]
    rows, cols, indices, flags = Int[], Int[], Int[], BitVector()
    rowtree, coltree = side === :left ? (cluster, free) : (free, cluster)
    rr, cc = ACA._clusterranges(rowtree), ACA._clusterranges(coltree)
    reference = zeros(
        T, length(rr[H2Trees.root(rowtree)]), length(cc[H2Trees.root(coltree)])
    )
    for (j, f) in enumerate(freeclusters)
        choice = pattern === :mixed ? (:root, :children, :leaves)[mod1(j, 3)] : pattern
        nodes = if choice === :root
            [root]
        elseif choice === :children
            children
        else
            diagonal
        end
        for (i, node) in enumerate(nodes)
            row, col = side === :left ? (node, f) : (f, node)
            isfar = kind === :far || kind === :mixed && isodd(i + j)
            if isfar
                U, V = randn(rng, T, length(rr[row]), 2), randn(rng, T, 2, length(cc[col]))
                reference[rr[row], cc[col]] = U * V
                push!(far, ACA.LowRankMatrix(U, V))
            else
                D = randn(rng, T, length(rr[row]), length(cc[col]))
                reference[rr[row], cc[col]] = D
                push!(near, D)
            end
            push!(rows, row)
            push!(cols, col)
            push!(flags, isfar)
            push!(indices, isfar ? length(far) : length(near))
        end
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
        if H.workspace.rowperm isa ACA.IdentityPermutation
            ACA.PermuteSpaceInPlace()
        else
            ACA.PreserveSpaceOrder()
        end,
        size(reference),
        1e-10,
    )
    return ACA._TreeView(core), reference
end

# Serial execution of a solve plan; returns the truncation status.
solve_test_status(scratch) = AdaptiveCrossApproximation._ProductStatus(
    true, false, zero(real(eltype(scratch.left))), 0
)

function solve_test_run!(
    P,
    scratch,
    side;
    atol=0,
    rtol=1e-6,
    maxrank=typemax(Int),
    status=solve_test_status(scratch),
)
    ACA = AdaptiveCrossApproximation
    for step in eachindex(P.entries)
        ACA._executesolveoperation!(P, step, scratch, status, (; atol, rtol, maxrank))
    end
    return status
end

function solve_test_dense(W)
    D = zeros(eltype(W), size(W))
    csr = W.csr
    for k in eachindex(csr.storageidx)
        rr, cc = AdaptiveCrossApproximation._productranges(W, k)
        D[rr, cc] =
            csr.isfar[k] ? Matrix(W.far[csr.storageidx[k]]) : W.near[csr.storageidx[k]]
    end
    return D
end

function solve_test_snapshot(W)
    return (copy.(W.near), [(copy(R.U), copy(R.V)) for R in W.far])
end
function solve_test_restore!(W, snapshot)
    foreach(copyto!, W.near, snapshot[1])
    for (R, (U, V)) in zip(W.far, snapshot[2])
        R.U, R.V = copy(U), copy(V)
    end
    return W
end

function solve_test_bytes(P, scratch, side)
    status = solve_test_status(scratch)
    solve_test_run!(P, scratch, side; status)
    return @allocated solve_test_run!(P, scratch, side; status)
end

@testset "solveplan.jl" begin
    @testset "_SolvePlan / HMatrixTriangular (unary selections)" begin
        ACA = AdaptiveCrossApproximation
        rng = MersenneTwister(19834)
        # Deliberately oversized root geometry creates a single occupied child
        # with exactly the same indices, followed by more unary geometric levels.
        points = [SVector(0.80 + 0.001i, 0.80) for i in 1:8]
        cluster = H2Trees.TwoNTree(SVector(0.0, 0.0), points, 1.0, 0.001; minvalues=2)
        root = H2Trees.root(cluster)
        @test length(collect(H2Trees.children(cluster, root))) == 1
        child = first(H2Trees.children(cluster, root))
        ranges = ACA._clusterranges(cluster)
        @test ranges[root] == ranges[child]
        tree, nodes = H2Trees.BlockTree(cluster, cluster), H2Trees.numberofnodes(cluster)
        for T in (Float64, ComplexF64)
            csr, csc = ACA._blockindices(
                [root], [root], falses(1), falses(1), [1], nodes, nodes
            )
            core = ACA.HMatrixCore{T}(
                tree,
                [randn(rng, T, 8, 8) + 8I],
                ACA.LowRankMatrix{T}[],
                csr,
                csc,
                ACA.PreserveSpaceOrder(),
                (8, 8),
                1e-10,
            )
            W = ACA._TreeView(core)
            F = ACA.leaflu!(only(W.near); node=root)
            for uplo in (:lower, :upper)
                H = ACA.HMatrixTriangular(W, [F]; uplo)
                selected = ACA.HMatrixTriangular(H; node=child)
                @test selected.node == root
                triangle = if uplo === :lower
                    UnitLowerTriangular(F.factors)
                else
                    UpperTriangular(F.factors)
                end
                plain = randn(rng, T, 8, 3)
                @test ACA._trisolve!(
                    copy(plain), selected, ACA.BlockScratch{T}(), Val(:N), Val(:left)
                ) ≈ triangle \ plain
                for side in (:left, :right), far in (false, true)
                    row, col = side === :left ? (child, root) : (root, child)
                    bcsr, bcsc = ACA._blockindices(
                        [row], [col], BitVector([far]), falses(1), [1], nodes, nodes
                    )
                    near = far ? Matrix{T}[] : [randn(rng, T, 8, 8)]
                    factors = if far
                        [ACA.LowRankMatrix(randn(rng, T, 8, 2), randn(rng, T, 2, 8))]
                    else
                        ACA.LowRankMatrix{T}[]
                    end
                    B = ACA._TreeView(
                        ACA.HMatrixCore{T}(
                            tree,
                            near,
                            factors,
                            bcsr,
                            bcsc,
                            ACA.PreserveSpaceOrder(),
                            (8, 8),
                            1e-10,
                        ),
                    )
                    original = far ? Matrix(only(B.far)) : copy(only(B.near))
                    dense = triangle
                    P = ACA._SolvePlan(H, ACA.blockview(B); side)
                    info = solve_test_run!(P, ACA.BlockScratch{T}(), side)
                    @test info.flushes == 0
                    actual = far ? Matrix(only(B.far)) : only(B.near)
                    @test actual ≈ (side === :left ? dense \ original : original / dense)
                end
            end
        end
    end

    @testset "_SolvePlan / _executesolveoperation! / _pivot!" begin
        ACA = AdaptiveCrossApproximation
        rng = MersenneTwister(9138)
        oldthreads = BLAS.get_num_threads()
        try
            BLAS.set_num_threads(1)
            for T in (Float64, ComplexF64, Float32, ComplexF32)
                W, leaves = triangular_test_fixture(rng, T)
                L, U = triangular_test_dense(W)
                factorcopy = solve_test_snapshot(W)
                tolerance = 2000eps(real(T))
                @testset "$T" begin
                    for uplo in (:lower, :upper), side in (:left, :right)
                        H = ACA.HMatrixTriangular(W, leaves; uplo)
                        dense =
                            uplo === :lower ? UnitLowerTriangular(L) : UpperTriangular(U)
                        B, original = solve_test_fixture(rng, T, H; side)
                        snapshot = solve_test_snapshot(B)
                        P = ACA._SolvePlan(H, ACA.blockview(B); side, batchrank=5)
                        @test length(P.entries) == length(B.csr)
                        expected = side === :left ? dense \ original : original / dense
                        scratch = ACA.BlockScratch{T}()
                        # Repeated execution reuses the plan and scratch.
                        for _ in 1:2
                            solve_test_restore!(B, snapshot)
                            info = solve_test_run!(P, scratch, side; rtol=0, atol=0)
                            actual = solve_test_dense(B)
                            @test info.converged
                            @test !info.rank_limited
                            @test actual ≈ expected rtol=tolerance atol=tolerance
                            residual = if side === :left
                                dense * actual - original
                            else
                                actual * dense - original
                            end
                            @test norm(residual) <=
                                tolerance * (norm(dense)*norm(actual) + norm(original))
                            @test solve_test_snapshot(W) == factorcopy
                        end
                        # Explicit permutations roundtrip without changing rank;
                        # pivots only apply to left-sided solves.
                        side === :left || continue
                        solve_test_restore!(B, snapshot)
                        permutation = Matrix{T}(I, size(H, 1), size(H, 1))
                        ACA._pivot!(permutation, H)
                        ranks = rank.(B.far)
                        @test ACA._pivot!(P) === P.rhs
                        @test solve_test_dense(B) ≈ permutation * original rtol=tolerance
                        ACA._pivot!(P; inverse=true)
                        @test solve_test_dense(B) ≈ original rtol=tolerance
                        @test rank.(B.far) == ranks
                    end
                end
            end

            @testset "Complete coarse low-rank owners" begin
                for T in (Float64, ComplexF64), side in (:left, :right)
                    W, leaves = triangular_test_fixture(
                        rng, T; coarsened=true, multidimensional=true, n=49
                    )
                    H = ACA.HMatrixTriangular(W, leaves; uplo=:upper)
                    _, U = triangular_test_dense(W)
                    dense = UpperTriangular(U)
                    B, original = solve_test_fixture(
                        rng, T, H; side, pattern=:root, kind=:far, freefront=:root
                    )
                    R = only(B.far)
                    before = (R.U, R.V)
                    P, scratch = ACA._SolvePlan(H, ACA.blockview(B); side),
                    ACA.BlockScratch{T}()
                    info = solve_test_run!(P, scratch, side)
                    @test isempty(P.products)
                    @test info.flushes == 0
                    @test rank(R) == 2
                    @test R.U === before[1] && R.V === before[2]
                    expected = side === :left ? dense \ original : original / dense
                    @test solve_test_dense(B) ≈ expected rtol=1e-11
                    # The factor not participating in the solve retains its values.
                    activeleft = side === :left
                    untouched = copy(activeleft ? R.V : R.U)
                    solve_test_run!(P, scratch, side)
                    @test (activeleft ? R.V : R.U) == untouched
                    @test solve_test_bytes(P, scratch, side) == 0
                    R.U, R.V = zeros(T, size(R.U, 1), 0), zeros(T, 0, size(R.V, 2))
                    @test solve_test_run!(P, scratch, side).flushes == 0
                    @test iszero(norm(solve_test_dense(B)))
                end
            end

            @testset "Solved physical regions are never revisited" begin
                for side in (:left, :right), uplo in (:lower, :upper)
                    W, leaves = triangular_test_fixture(
                        rng, ComplexF64; multidimensional=true, n=49
                    )
                    H = ACA.HMatrixTriangular(W, leaves; uplo)
                    B, _ = solve_test_fixture(
                        rng, ComplexF64, H; side, pattern=:leaves, kind=:far
                    )
                    P = ACA._SolvePlan(H, ACA.blockview(B); side, batchrank=3)
                    scratch = ACA.BlockScratch{ComplexF64}()
                    status = ACA._ProductStatus(true, false, 0.0, 0)
                    options = (; atol=0, rtol=1e-8, maxrank=2)
                    solved = Dict{Int,Tuple{Matrix{ComplexF64},Matrix{ComplexF64}}}()
                    for step in eachindex(P.entries)
                        ACA._executesolveoperation!(P, step, scratch, status, options)
                        R = B.far[B.csr.storageidx[P.entries[step]]]
                        solved[P.entries[step]] = (copy(R.U), copy(R.V))
                        for (k, (U, V)) in solved
                            previous = B.far[B.csr.storageidx[k]]
                            @test previous.U == U && previous.V == V
                        end
                    end
                end
            end

            @testset "Subtree plans and convenience methods" begin
                for side in (:left, :right), uplo in (:lower, :upper)
                    W, leaves = triangular_test_fixture(rng, Float64)
                    H = ACA.HMatrixTriangular(W, leaves; uplo)
                    node = last(collect(H2Trees.children(H2Trees.testtree(W.tree), H.node)))
                    selected = ACA.HMatrixTriangular(H; node)
                    B, original = solve_test_fixture(rng, Float64, H; side, pattern=:leaves)
                    range = selected.range
                    viewB = if side === :left
                        ACA.blockview(B, range, 1:size(B, 2))
                    else
                        ACA.blockview(B, 1:size(B, 1), range)
                    end
                    before = solve_test_dense(B)
                    L, U = triangular_test_dense(W)
                    triangle = if uplo === :lower
                        UnitLowerTriangular(L[range, range])
                    else
                        UpperTriangular(U[range, range])
                    end
                    expected = if side === :left
                        triangle \ original[range, :]
                    else
                        original[:, range] / triangle
                    end
                    scratch = ACA.BlockScratch{Float64}()
                    info = solve_test_run!(
                        ACA._SolvePlan(selected, viewB; side), scratch, side; rtol=0
                    )
                    @test info.converged
                    actual = solve_test_dense(B)
                    @test (side === :left ? actual[range, :] : actual[:, range]) ≈ expected
                    outside = setdiff(1:size(H, 1), range)
                    @test (side === :left ? actual[outside, :] : actual[:, outside]) ==
                        (side === :left ? before[outside, :] : before[:, outside])
                end
            end

            @testset "Invalid plans" begin
                W, leaves = triangular_test_fixture(rng, Float64; coarsened=true)
                H = ACA.HMatrixTriangular(W, leaves)
                B, _ = solve_test_fixture(rng, Float64, H; pattern=:leaves)
                snapshot = solve_test_snapshot(B)
                @test_throws ArgumentError ACA._SolvePlan(
                    H, ACA.blockview(B); side=:invalid
                )
                @test_throws ArgumentError ACA._SolvePlan(H, ACA.blockview(B); batchrank=0)
                # Views cannot cut a coarse physical block, including on the free axis.
                coarse, _ = solve_test_fixture(
                    rng, Float64, H; pattern=:root, kind=:far, freefront=:root
                )
                @test_throws ArgumentError ACA._SolvePlan(
                    H, ACA.blockview(coarse, 1:size(H, 1), 2:size(coarse, 2))
                )
                node = first(H2Trees.children(H2Trees.testtree(W.tree), H.node))
                sub = ACA.HMatrixTriangular(H; node)
                @test_throws ArgumentError ACA._SolvePlan(
                    sub, ACA.blockview(coarse, sub.range, 1:size(coarse, 2))
                )
                # A finer RHS frontier would require splitting a dense factor leaf.
                fine, _ = product_test_fixture(
                    rng,
                    Float64,
                    H2Trees.testtree(W.tree),
                    H2Trees.testtree(W.tree);
                    rowfront=:leaves,
                    colfront=:root,
                )
                @test_throws ArgumentError ACA._SolvePlan(H, ACA.blockview(fine))
                @test solve_test_snapshot(B) == snapshot
            end

            @testset "Ordering, empty selections and changing ranks" begin
                W, leaves = triangular_test_fixture(
                    rng, ComplexF64; ordering=ACA.PermuteSpaceInPlace()
                )
                H = ACA.HMatrixTriangular(W, leaves; uplo=:upper)
                for side in (:left, :right)
                    B, original = solve_test_fixture(
                        rng, ComplexF64, H; side, pattern=:leaves
                    )
                    snapshot = solve_test_snapshot(B)
                    @test (side === :left ? B.rowperm : B.colperm) isa
                        ACA.IdentityPermutation
                    P = ACA._SolvePlan(H, ACA.blockview(B); side, batchrank=4)
                    scratch = ACA.BlockScratch{ComplexF64}()
                    _, U = triangular_test_dense(W)
                    expected = if side === :left
                        UpperTriangular(U) \ original
                    else
                        original / UpperTriangular(U)
                    end
                    @test solve_test_run!(P, scratch, side; rtol=0).converged
                    @test solve_test_dense(B) ≈ expected
                    solve_test_restore!(B, snapshot)
                    for R in B.far
                        R.U, R.V = zeros(ComplexF64, size(R.U, 1), 0),
                        zeros(ComplexF64, 0, size(R.V, 2))
                    end
                    zeroed = solve_test_dense(B)
                    expected = if side === :left
                        UpperTriangular(U) \ zeroed
                    else
                        zeroed / UpperTriangular(U)
                    end
                    @test solve_test_run!(P, scratch, side; rtol=0).converged
                    @test solve_test_dense(B) ≈ expected
                    solve_test_restore!(B, snapshot)
                    emptyrhs = if side === :left
                        ACA.blockview(B, 1:size(B, 1), 1:0)
                    else
                        ACA.blockview(B, 1:0, 1:size(B, 2))
                    end
                    emptyplan = ACA._SolvePlan(H, emptyrhs; side)
                    @test isempty(emptyplan.entries)
                    @test solve_test_run!(emptyplan, scratch, side).flushes == 0
                    @test solve_test_snapshot(B) == snapshot
                    # A completely unrelated solve-axis tree is not silently aligned.
                    foreign, foreignleaves = triangular_test_fixture(
                        rng, ComplexF64; ordering=ACA.PermuteSpaceInPlace()
                    )
                    @test_throws ArgumentError ACA._SolvePlan(
                        ACA.HMatrixTriangular(foreign, foreignleaves),
                        ACA.blockview(B);
                        side,
                    )
                end
            end

            @testset "Local truncation status and reset" begin
                W, leaves = triangular_test_fixture(rng, ComplexF64)
                H = ACA.HMatrixTriangular(W, leaves)
                for side in (:left, :right)
                    B, _ = solve_test_fixture(
                        rng, ComplexF64, H; side, pattern=:leaves, kind=:far
                    )
                    snapshot = solve_test_snapshot(B)
                    P = ACA._SolvePlan(H, ACA.blockview(B); side)
                    scratch = ACA.BlockScratch{ComplexF64}()
                    info = solve_test_run!(P, scratch, side; maxrank=0, rtol=0)
                    @test !info.converged && info.rank_limited
                    @test info.flushes > 0 && info.max_block_error > 0
                    solve_test_restore!(B, snapshot)
                    info = solve_test_run!(P, scratch, side; rtol=0)
                    @test info.converged && !info.rank_limited
                    @test info.max_block_error == 0
                end
            end
        finally
            BLAS.set_num_threads(oldthreads)
        end
    end
end
