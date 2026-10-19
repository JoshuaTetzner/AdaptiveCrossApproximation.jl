using AdaptiveCrossApproximation
using H2Trees
using LinearAlgebra
using Random
using StaticArrays
using Test
Random.seed!(1234)

@testset "hmatrix.jl" begin
    @testset "HMatrix" begin
        tpts = [SVector(rand(), rand(), 0.0) for i in 1:200]
        farpts = [SVector(rand() + 3.0, rand(), 0.0) for i in 1:210]
        farnearpts = [SVector(rand() + 2.0, rand(), 0.0) for i in 1:210]

        for mesh in [(tpts, tpts), (tpts, farpts), (tpts, farnearpts)]
            for spaceordering in [
                AdaptiveCrossApproximation.PermuteSpaceInPlace(),
                AdaptiveCrossApproximation.PreserveSpaceOrder(),
            ]
                for tol in [1e-2, 1e-4, 1e-6]
                    tree = buildtree(
                        mesh[1],
                        mesh[2];
                        builder=BlockTreeBuilder(;
                            test=TwoNTreeBuilder(; minhalfsize=1 / 2^10, minvalues=100),
                            trial=TwoNTreeBuilder(; minhalfsize=1 / 2^10, minvalues=100),
                        ),
                    )

                    mat = AdaptiveCrossApproximation.HMatrix(
                        fct, mesh[1], mesh[2], tree; spaceordering=spaceordering, tol=tol
                    )
                    A = [fct(x, y) for x in mesh[1], y in mesh[2]]
                    @test norm(Matrix(mat) - A) / norm(A) < tol
                    x = rand(eltype(mat), size(mat, 2))
                    @test norm(mat * x - A * x) / norm(A * x) < tol
                    x = rand(eltype(mat), size(mat, 1))
                    @test norm(transpose(mat) * x - transpose(A) * x) /
                          norm(transpose(A) * x) < tol
                    @test norm(adjoint(mat) * x - adjoint(A) * x) / norm(adjoint(A) * x) <
                        tol

                    # Float32 tolerances below 1e-5 approach roundoff.
                    tol32 = max(tol, 1e-5)
                    mat = AdaptiveCrossApproximation.HMatrix(
                        fct32,
                        mesh[1],
                        mesh[2],
                        tree;
                        spaceordering=spaceordering,
                        tol=tol32,
                    )
                    @test eltype(mat) == Float32
                    A = [fct32(x, y) for x in mesh[1], y in mesh[2]]
                    @test norm(Matrix(mat) - A) / norm(A) < tol32

                    x = rand(eltype(mat), size(mat, 2))
                    y = mat * x
                    @test eltype(y) == Float32
                    @test norm(y - A * x) / norm(A * x) < tol32

                    x = rand(eltype(mat), size(mat, 1))
                    y = transpose(mat) * x
                    @test eltype(y) == Float32
                    @test norm(y - transpose(A) * x) / norm(transpose(A) * x) < tol32
                    y = adjoint(mat) * x
                    @test eltype(y) == Float32
                    @test norm(y - adjoint(A) * x) / norm(adjoint(A) * x) < tol32
                end
            end
        end
    end

    @testset "HMatrixCore" begin
        points = [SVector(rand(), rand(), 0.0) for _ in 1:40]
        tree = H2Trees.buildtree(
            points,
            points;
            builder=H2Trees.BlockTreeBuilder(;
                test=H2Trees.TwoNTreeBuilder(; minhalfsize=1 / 2^10, minvalues=20),
                trial=H2Trees.TwoNTreeBuilder(; minhalfsize=1 / 2^10, minvalues=20),
            ),
        )
        mat = AdaptiveCrossApproximation.HMatrix(
            fct,
            points,
            points,
            tree;
            spaceordering=AdaptiveCrossApproximation.PreserveSpaceOrder(),
            tol=1e-4,
        )
        near = AdaptiveCrossApproximation.nearmatrix(mat)
        far = AdaptiveCrossApproximation.farmatrix(mat)
        rowtree = H2Trees.testtree(tree)
        coltree = H2Trees.trialtree(tree)
        rowleaves = [
            copy(H2Trees.values(rowtree, leaf)) for leaf in H2Trees.leaves(rowtree)
        ]
        colleaves = [
            copy(H2Trees.values(coltree, leaf)) for leaf in H2Trees.leaves(coltree)
        ]
        index = AdaptiveCrossApproximation._TreeIndex(mat)
        view = AdaptiveCrossApproximation._TreeView(mat)
        storedentries =
            sum(length, mat.near) +
            sum(length(block.U) + length(block.V) for block in mat.far; init=0)

        @test mat isa AdaptiveCrossApproximation.HMatrixCore
        @test mat.tol == 1e-4
        @test index.rowperm == H2Trees.values(rowtree, H2Trees.root(rowtree))
        @test index.colperm == H2Trees.values(coltree, H2Trees.root(coltree))
        @test rowleaves ==
            [H2Trees.values(rowtree, leaf) for leaf in H2Trees.leaves(rowtree)]
        @test colleaves ==
            [H2Trees.values(coltree, leaf) for leaf in H2Trees.leaves(coltree)]
        @test all(view.near[i] === mat.near[i] for i in eachindex(mat.near))
        @test all(view.far[i] === mat.far[i] for i in eachindex(mat.far))
        @testset "copy" begin
            copied = copy(mat)
            @test copied.tol == mat.tol && copied.csr === mat.csr
            @test all(copied.near[i] !== mat.near[i] for i in eachindex(mat.near))
            @test all(
                copied.far[i].U !== mat.far[i].U && copied.far[i].V !== mat.far[i].V for
                i in eachindex(mat.far)
            )
            @test Matrix(copied) == Matrix(mat)
        end

        @testset "nearmatrix / farmatrix" begin
            @test Matrix(near) + Matrix(far) ≈ Matrix(mat)
        end
        @testset "nnz" begin
            @test AdaptiveCrossApproximation.nnz(mat) == storedentries
        end
        @testset "storage" begin
            @test AdaptiveCrossApproximation.storage(mat) ==
                storedentries * sizeof(eltype(mat)) * 1e-9
        end
        @test length(mat.csr) == length(mat.near) + length(mat.far)
        @test sort(mat.csc.csrmap) == collect(1:length(mat.csr))
        @test count(!, mat.csr.isfar) == length(mat.near)
        @test count(identity, mat.csr.isfar) == length(mat.far)
    end
end
