using AdaptiveCrossApproximation
using H2Trees
using LinearAlgebra
using StaticArrays
using Test
using BEAST
using CompScienceMeshes
using ParallelKMeans

@testset "matvec.jl" begin
    @testset "MatVecWorkspace / mul!" begin
        ACA = AdaptiveCrossApproximation
        for symmetric in (false, true),
            ordering in (ACA.PreserveSpaceOrder(), ACA.PermuteSpaceInPlace())

            @testset "symmetric=$symmetric, ordering=$(typeof(ordering))" begin
                (; core, reference, rowperm, colperm) = arithmetic_workspace_fixture(
                    ordering; symmetric
                )
                rowtree, coltree = H2Trees.testtree(core.tree), H2Trees.trialtree(core.tree)
                leavesbefore = (
                    [copy(H2Trees.values(rowtree, n)) for n in H2Trees.leaves(rowtree)],
                    [copy(H2Trees.values(coltree, n)) for n in H2Trees.leaves(coltree)],
                )
                index = @inferred ACA._TreeIndex(core)

                @testset "mul!" begin
                    x = ComplexF64[complex(i, -i / 3) for i in 1:size(core, 2)]
                    xt = ComplexF64[complex(-i / 2, i) for i in 1:size(core, 1)]
                    X = reshape(ComplexF64.(1:(3 * size(core, 2))), size(core, 2), 3)
                    for ntasks in (1, 2)
                        A = ACA.MatVecWorkspace(core; ntasks)
                        @test A * x ≈ reference * x
                        @test transpose(A) * xt ≈ transpose(reference) * xt
                        @test A' * xt ≈ reference' * xt
                        @test A * X ≈ reference * X
                        y = ones(ComplexF64, size(core, 1))
                        @test mul!(y, A, x, 2, -1) ≈ 2 * reference * x .- 1
                        B = ACA.MatVecWorkspace(A)
                        @test B.z !== A.z && B.offsets !== A.offsets
                        @test B * x ≈ reference * x
                    end
                end
            end
        end
    end

    k = 2.4567799554075624 + 0.0im

    circ = CompScienceMeshes.meshcircle(1.0, 0.025)
    X = BEAST.lagrangecx(circ; order=2)
    tree = BlockTree(
        KMeansTree(X.pos; builder=KMeansTreeBuilder(; numberofclusters=2, minvalues=100)),
        KMeansTree(X.pos; builder=KMeansTreeBuilder(; numberofclusters=2, minvalues=100)),
    )

    op = Helmholtz2D.doublelayer(; wavenumber=k)
    quadstrat = BEAST.DoubleNumSauterQstrat(2, 3, 0, 4, 3, 4)
    DLop = AdaptiveCrossApproximation.HMatrix(
        op,
        X,
        X,
        tree;
        tol=1e-3,
        nearmatrixdata=BEAST.DoubleNumSauterQstrat(2, 3, 0, 4, 3, 4),
        farmatrixdata=BEAST.DoubleNumSauterQstrat(2, 3, 0, 4, 3, 4),
        spaceordering=AdaptiveCrossApproximation.PreserveSpaceOrder(),
    )
    Iop = Matrix(assemble(BEAST.Identity(), X, X))

    DL = DLop + (-0.5 .* Iop)
    DL2_op = assemble(op, X, X; quadstrat=quadstrat)
    DL2 = DL2_op + (-0.5 .* Iop)
    ##
    x = rand(eltype(DLop), size(DLop, 2))
    @testset "mul! (complex BEAST matrices)" begin
        @test norm(DLop * x - DL2_op * x) / norm(DL2_op * x) < 1e-3
        @test norm(adjoint(DLop) * x - adjoint(DL2_op) * x) / norm(adjoint(DL2_op) * x) <
            1e-3
        @test norm(DL * x - DL2 * x) / norm(DL2 * x) < 1e-3
    end
end
