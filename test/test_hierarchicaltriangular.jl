using AdaptiveCrossApproximation
using H2Trees
using LinearAlgebra
using Random
using StaticArrays
using Test

# Exact factors and independent dense triangular references; no ACA or
# hierarchical LU is used to manufacture a supposedly correct test answer.
function triangular_test_fixture(
    rng,
    T;
    coarsened=false,
    n=29,
    multidimensional=false,
    rank=3,
    ordering=AdaptiveCrossApproximation.PreserveSpaceOrder(),
    factorize=true,
)
    ACA = AdaptiveCrossApproximation
    points = [
        if multidimensional
            SVector(Float64(mod(i, 7)), Float64(i ÷ 7))
        else
            SVector(Float64(i)^2, 0.0)
        end for i in randperm(rng, n)
    ]
    cluster = H2Trees.buildtree(
        points; builder=H2Trees.TwoNTreeBuilder(; minvalues=3, minhalfsize=0.001)
    )
    tree = H2Trees.BlockTree(cluster, cluster)
    ordering(tree, points, points)
    ranges = ACA._clusterranges(cluster)
    root = H2Trees.root(cluster)
    coarse = coarsened ? first(H2Trees.children(cluster, root)) : 0
    near, far = Matrix{T}[], ACA.LowRankMatrix{T}[]
    rows, cols, indices = Int[], Int[], Int[]
    flags = BitVector()
    function terminal(row, col)
        rr, cc = ranges[row], ranges[col]
        diagonal = row == col
        isfar = !diagonal && isodd(row + col)
        stop =
            isfar ||
            diagonal && row == coarse ||
            H2Trees.isleaf(cluster, row) ||
            H2Trees.isleaf(cluster, col)
        if !stop
            for r in H2Trees.children(cluster, row), c in H2Trees.children(cluster, col)
                terminal(r, c)
            end
            return nothing
        end
        if isfar
            U, V = randn(rng, T, length(rr), rank), randn(rng, T, rank, length(cc))
            push!(far, ACA.LowRankMatrix(U, V))
        else
            D = randn(rng, T, length(rr), length(cc))
            if diagonal
                D += 10length(rr)*I
                D = D[circshift(collect(1:length(rr)), 1), :]
            end
            push!(near, D)
        end
        index = isfar ? length(far) : length(near)
        push!(rows, row)
        push!(cols, col)
        push!(indices, index)
        push!(flags, isfar)
        return nothing
    end
    terminal(root, root)
    csr, csc = ACA._blockindices(
        rows,
        cols,
        flags,
        falses(length(flags)),
        indices,
        H2Trees.numberofnodes(cluster),
        H2Trees.numberofnodes(cluster),
    )
    core = ACA.HMatrixCore{T}(tree, near, far, csr, csc, ordering, (n, n), 1e-10)
    factorize || return core, nothing
    W = ACA._TreeView(core)
    leaves = [
        ACA.leaflu!(W.near[csr.storageidx[k]]; node=W.entryrows[k]) for
        k in eachindex(csr.storageidx) if W.entryrows[k] == csr.colnodes[k]
    ]
    return W, leaves
end

function triangular_test_dense(W)
    L, U = zeros(eltype(W), size(W)), zeros(eltype(W), size(W))
    csr = W.csr
    for k in eachindex(csr.storageidx)
        row, col = W.entryrows[k], csr.colnodes[k]
        rr, cc = W.rowranges[row], W.colranges[col]
        logical =
            csr.isfar[k] ? Matrix(W.far[csr.storageidx[k]]) : W.near[csr.storageidx[k]]
        if row == col
            L[rr, cc] = UnitLowerTriangular(logical)
            U[rr, cc] = UpperTriangular(logical)
        elseif first(rr) > first(cc)
            L[rr, cc] = logical
        else
            U[rr, cc] = logical
        end
    end
    return L, U
end

# Transposed factors are a solve option of the internal kernel.
struct TriangularTestTranspose{H}
    parent::H
end
triangular_test_op(H, op) = op === :N ? H : TriangularTestTranspose(H)
triangular_test_parts(H::AdaptiveCrossApproximation.HMatrixTriangular) = (H, Val(:N))
triangular_test_parts(H::TriangularTestTranspose) = (H.parent, Val(:T))

function triangular_test_solve!(B, H, scratch, side)
    F, op = triangular_test_parts(H)
    if side === :left
        return AdaptiveCrossApproximation._trisolve!(B, F, scratch, op, Val(:left))
    end
    return AdaptiveCrossApproximation._trisolve!(B, F, scratch, op, Val(:right))
end
function triangular_test_bytes(B, H, scratch, side)
    triangular_test_solve!(B, H, scratch, side)
    return @allocated triangular_test_solve!(B, H, scratch, side)
end
function triangular_test_pivotbytes(B, H)
    AdaptiveCrossApproximation._pivot!(B, H)
    return @allocated AdaptiveCrossApproximation._pivot!(B, H)
end

@testset "Hierarchical triangular solves" begin
    ACA = AdaptiveCrossApproximation
    rng = MersenneTwister(12281)
    for T in (Float64, ComplexF64, Float32, ComplexF32), coarsened in (false, true)
        tol = 2000eps(real(T))
        W, leaves = triangular_test_fixture(rng, T; coarsened)
        L, U = triangular_test_dense(W)
        before = (copy.(W.near), [(copy(R.U), copy(R.V)) for R in W.far])
        @test length(
            unique(
                length.(
                    W.rowranges[findall(!iszero, ACA.HMatrixTriangular(W, leaves).leafmap)]
                ),
            ),
        ) >= 2
        for uplo in (:lower, :upper)
            H = @inferred ACA.HMatrixTriangular(W, reverse(leaves); uplo)
            @test H.workspace === W
            @test issuccess(H)
            @test size(H) == size(W)
            reference = uplo === :lower ? L : U
            for trans in (:N, :T), side in (:left, :right)
                opH = triangular_test_op(H, trans)
                triangle = trans === :N ? reference : transpose(reference)
                scratch = ACA.BlockScratch{T}()
                for q in (0, 1, 5, 2, 19)
                    shape = side === :left ? (size(H, 1), q) : (q, size(H, 1))
                    B = randn(rng, T, shape...)
                    original = copy(B)
                    expected = side === :left ? triangle \ B : B / triangle
                    @test triangular_test_solve!(B, opH, scratch, side) === B
                    @test B ≈ expected rtol=tol atol=tol*max(norm(expected), 1)
                    residual =
                        side === :left ? triangle * B - original : B * triangle - original
                    @test norm(residual) <= tol*max(norm(triangle)*norm(B), 1)
                end
                # Padded RHS views, and stable warmed capacity across calls.
                parent = fill(T(29), 35, 35)
                B = side === :left ? view(parent, 3:31, 2:4) : view(parent, 2:4, 3:31)
                original = randn(rng, T, size(B))
                copyto!(B, original)
                expected = side === :left ? triangle \ original : original / triangle
                triangular_test_solve!(B, opH, scratch, side)
                @test B ≈ expected rtol=tol atol=tol*max(norm(expected), 1)
                @test all(==(T(29)), view(parent, 1:1, :))
                pointerbefore = pointer(scratch.small)
                bytes = triangular_test_bytes(B, opH, scratch, side)
                @test bytes == 0
                @test pointer(scratch.small) == pointerbefore
            end
        end
        @test W.near == before[1]
        @test [(R.U, R.V) for R in W.far] == before[2]
        H = ACA.HMatrixTriangular(W, leaves)
        P = Matrix{T}(I, 29, 29)
        for F in leaves
            AdaptiveCrossApproximation._leafpivots!(
                view(P, W.rowranges[F.node], :), F.ipiv, false
            )
        end
        for inverse in (false, true)
            B = randn(rng, T, 29, 29)
            original = copy(B)
            ACA._pivot!(B, H; inverse)
            @test B == (inverse ? transpose(P) : P) * original
            ACA._pivot!(B, H; inverse=(!inverse))
            @test B == original
            triangular_test_pivotbytes(B, H)
            @test triangular_test_pivotbytes(B, H) == 0
        end
        # A synthetic full factorization with the SAME leaf permutations.
        A = transpose(P) * L * U
        B = randn(rng, T, 29, 3)
        original = copy(B)
        ACA._pivot!(B, H)
        triangular_test_solve!(B, H, ACA.BlockScratch{T}(), :left)
        triangular_test_solve!(
            B, ACA.HMatrixTriangular(W, leaves; uplo=:upper), ACA.BlockScratch{T}(), :left
        )
        @test A * B ≈ original rtol=tol atol=tol*norm(A)*norm(B)
    end

    @testset "Validation and changing ranks" begin
        W, leaves = triangular_test_fixture(rng, ComplexF64)
        H = ACA.HMatrixTriangular(W, leaves)
        scratch = ACA.BlockScratch{ComplexF64}()
        B = ones(ComplexF64, 29, 3)
        @test_throws ArgumentError ACA.HMatrixTriangular(W, leaves; uplo=:bad)
        @test_throws ArgumentError ACA.HMatrixTriangular(W, leaves[2:end])
        @test_throws ArgumentError ACA.HMatrixTriangular(W, [leaves; leaves[1]])
        foreign = copy(leaves)
        foreign[1] = ACA.leaflu!(copy(leaves[1].factors))
        @test_throws ArgumentError ACA.HMatrixTriangular(W, foreign)
        @test_throws DimensionMismatch triangular_test_solve!(
            ones(ComplexF64, 28, 3), H, scratch, :left
        )
        @test_throws DimensionMismatch triangular_test_solve!(
            ones(ComplexF64, 3, 28), H, scratch, :right
        )
        @test_throws ArgumentError triangular_test_solve!(
            view(ones(ComplexF64, 58, 3), 1:2:58, :), H, scratch, :left
        )
        scratch.small = vec(B)
        @test_throws ArgumentError triangular_test_solve!(B, H, scratch, :left)
        @test B == ones(29, 3)
        scratch = ACA.BlockScratch{ComplexF64}()
        leaves[end].info = 1
        @test !issuccess(H)
        @test_throws LeafLUFailure triangular_test_solve!(B, H, scratch, :left)
        @test B == ones(29, 3)
        leaves[end].info = 0
        leaves[end].ipiv[end] = 0
        @test_throws ArgumentError ACA._pivot!(B, H)
        @test B == ones(29, 3)
        leaves[end].ipiv[end] = length(leaves[end].ipiv)
        original = W.near[W.csr.storageidx[H.leafrecords[1]]]
        W.near[W.csr.storageidx[H.leafrecords[1]]] = copy(original)
        @test_throws ArgumentError triangular_test_solve!(B, H, scratch, :left)
        W.near[W.csr.storageidx[H.leafrecords[1]]] = original
        # Dynamic far ranks must be reread without rebuilding execution indices.
        for R in W.far
            R.U = zeros(ComplexF64, size(R.U, 1), 0)
            R.V = zeros(ComplexF64, 0, size(R.V, 2))
        end
        L, _ = triangular_test_dense(W)
        triangular_test_solve!(B, H, scratch, :left)
        @test L * B ≈ ones(29, 3)
    end

    @testset "Axis/diagonal invariants and multiway trees" begin
        W, leaves = triangular_test_fixture(rng, ComplexF64; multidimensional=true, n=49)
        tree = H2Trees.testtree(W.tree)
        @test length(collect(H2Trees.children(tree, H2Trees.root(tree)))) > 2
        for F in leaves
            F.node = 0 # Labels are optional; cached cluster associations are authoritative.
        end
        L, U = triangular_test_dense(W)
        for uplo in (:lower, :upper), side in (:left, :right), op in (:N, :T)
            H = ACA.HMatrixTriangular(W, leaves; uplo)
            A = uplo === :lower ? L : U
            opH = triangular_test_op(H, op)
            opA = op === :N ? A : transpose(A)
            B = randn(rng, ComplexF64, side === :left ? (49, 7) : (7, 49))
            expected = side === :left ? opA \ B : B / opA
            triangular_test_solve!(B, opH, ACA.BlockScratch{ComplexF64}(), side)
            @test B ≈ expected rtol=1e-10
        end
        H = ACA.HMatrixTriangular(W, leaves)
        B = randn(rng, ComplexF64, 49, 3)
        original = copy(B)
        ACA._pivot!(B, H)
        ACA._pivot!(B, H; inverse=true)
        @test B == original
        csr = W.csr
        malformed = ACA.BlockCSR(
            copy(csr.rowptr),
            copy(csr.colnodes),
            copy(csr.isfar),
            copy(csr.transposed),
            copy(csr.storageidx),
        )
        malformed.transposed[H.leafrecords[1]] = true
        ordering = ACA.PreserveSpaceOrder()
        badcore = ACA.HMatrixCore{ComplexF64}(
            W.tree, W.near, W.far, malformed, W.csc, ordering, size(W), 1e-10
        )
        @test_throws ArgumentError ACA._TreeView(badcore)
        rectangular = ACA.HMatrixCore{ComplexF64}(
            W.tree, W.near, W.far, csr, W.csc, ordering, (49, 48), 1e-10
        )
        @test_throws DimensionMismatch ACA.HMatrixTriangular(
            ACA._TreeView(rectangular), leaves
        )
        unrelated = ACA.HMatrixCore{ComplexF64}(
            H2Trees.BlockTree(tree, deepcopy(tree)),
            W.near,
            W.far,
            csr,
            W.csc,
            ordering,
            size(W),
            1e-10,
        )
        @test_throws ArgumentError ACA.HMatrixTriangular(ACA._TreeView(unrelated), leaves)
        ordered, orderedleaves = triangular_test_fixture(
            rng, Float64; ordering=ACA.PermuteSpaceInPlace()
        )
        @test ordered.rowperm isa ACA.IdentityPermutation
        lower, _ = triangular_test_dense(ordered)
        B = ones(29, 2)
        triangular_test_solve!(
            B,
            ACA.HMatrixTriangular(ordered, orderedleaves),
            ACA.BlockScratch{Float64}(),
            :left,
        )
        @test lower * B ≈ ones(29, 2)
    end

    @testset "Shared execution metadata for diagonal subtree solves" begin
        W, leaves = triangular_test_fixture(rng, ComplexF64; multidimensional=true, n=49)
        tree = H2Trees.testtree(W.tree)
        L, U = triangular_test_dense(W)
        children = collect(H2Trees.children(tree, H2Trees.root(tree)))
        for uplo in (:lower, :upper)
            master = ACA.HMatrixTriangular(W, leaves; uplo)
            for node in children
                H = ACA.HMatrixTriangular(master; node)
                rr = W.rowranges[node]
                @test H.panels === master.panels
                @test H.leafmap === master.leafmap
                @test H.leftptr === master.leftptr
                @test H.range == rr
                @test size(H) == (length(rr), length(rr))
                @test size(H, 3) == 1
                A = (uplo === :lower ? L : U)[rr, rr]
                for op in (:N, :T), side in (:left, :right)
                    opH = triangular_test_op(H, op)
                    opA = op === :N ? A : transpose(A)
                    B = randn(
                        rng, ComplexF64, side === :left ? (length(rr), 3) : (3, length(rr))
                    )
                    expected = side === :left ? opA \ B : B / opA
                    scratch = ACA.BlockScratch{ComplexF64}()
                    triangular_test_solve!(B, opH, scratch, side)
                    @test B ≈ expected rtol=1e-10
                    triangular_test_bytes(B, opH, scratch, side)
                    @test triangular_test_bytes(B, opH, scratch, side) == 0
                end
                P = Matrix{ComplexF64}(I, length(rr), length(rr))
                ACA._pivot!(P, H)
                @test P * transpose(P) == I
            end
            H = ACA.HMatrixTriangular(master; node=children[1])
            master.leaves[end].info = -1 # Unrelated trailing subtree not factorized yet.
            @test issuccess(H) && !issuccess(master)
            B = ones(ComplexF64, size(H, 1), 2)
            expected = (uplo === :lower ? L : U)[H.range, H.range] \ B
            triangular_test_solve!(B, H, ACA.BlockScratch{ComplexF64}(), :left)
            @test B ≈ expected
            unready = ACA.HMatrixTriangular(W, leaves; uplo, check=false)
            @test !issuccess(unready)
            @test_throws ArgumentError triangular_test_solve!(
                ones(ComplexF64, 49, 2), unready, ACA.BlockScratch{ComplexF64}(), :left
            )
            master.leaves[end].info = 0
            @test_throws BoundsError ACA.HMatrixTriangular(master; node=0)
            @test_throws ArgumentError ACA.HMatrixTriangular(H; node=children[end])
        end
        W, leaves = triangular_test_fixture(rng, Float64; coarsened=true)
        H = ACA.HMatrixTriangular(W, leaves)
        coarse = first(
            H2Trees.children(
                H2Trees.testtree(W.tree), H2Trees.root(H2Trees.testtree(W.tree))
            ),
        )
        cut = first(H2Trees.children(H2Trees.testtree(W.tree), coarse))
        @test_throws ArgumentError ACA.HMatrixTriangular(H; node=cut)
    end
end
