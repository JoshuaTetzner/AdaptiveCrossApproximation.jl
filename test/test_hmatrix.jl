using AdaptiveCrossApproximation
using H2Trees
using LinearAlgebra
using Random
using StaticArrays
using Test
Random.seed!(1234)

struct myfct end
Base.eltype(::myfct) = Float64
function (::myfct)(x, y)
    if x == y
        return 0.0
    else
        return inv(norm(x - y))
    end
end
fct = myfct()

struct myfct32 end
Base.eltype(::myfct32) = Float32
function (::myfct32)(x, y)
    if x == y
        return 0.0
    else
        return inv(norm(x - y))
    end
end
fct32 = myfct32()

@testset "H-Matrix" begin
    tpts = [SVector(rand(), rand(), 0.0) for i in 1:200]
    farpts = [SVector(rand() + 3.0, rand(), 0.0) for i in 1:210]
    farnearpts = [SVector(rand() + 2.0, rand(), 0.0) for i in 1:210]

    for mesh in [(tpts, tpts), (tpts, farpts), (tpts, farnearpts)]
        for spaceordering in [
            AdaptiveCrossApproximation.PermuteSpaceInPlace(),
            AdaptiveCrossApproximation.PreserveSpaceOrder(),
        ]
            for tol in [1e-2, 1e-4, 1e-6]
                tree = TwoNTree(
                    mesh[1], mesh[2], 1 / 2^10; testminvalues=100, trialminvalues=100
                )

                mat = AdaptiveCrossApproximation.HMatrix(
                    fct, mesh[1], mesh[2], tree; spaceordering=spaceordering, tol=tol
                )
                A = [fct(x, y) for x in mesh[1], y in mesh[2]]
                @test norm(Matrix(mat) - A) / norm(A) < tol
                x = rand(eltype(mat), size(mat, 2))
                @test norm(mat * x - A * x) / norm(A * x) < tol
                x = rand(eltype(mat), size(mat, 1))
                @test norm(transpose(mat) * x - transpose(A) * x) / norm(transpose(A) * x) <
                    tol
                @test norm(adjoint(mat) * x - adjoint(A) * x) / norm(adjoint(A) * x) < tol

                mat = AdaptiveCrossApproximation.HMatrix(
                    fct32, mesh[1], mesh[2], tree; spaceordering=spaceordering, tol=tol
                )
                @test eltype(mat) == Float32
                A = [fct32(x, y) for x in mesh[1], y in mesh[2]]
                @test norm(Matrix(mat) - A) / norm(A) < tol

                x = rand(eltype(mat), size(mat, 2))
                y = mat * x
                @test eltype(y) == Float32
                @test norm(y - A * x) / norm(A * x) < tol

                x = rand(eltype(mat), size(mat, 1))
                y = transpose(mat) * x
                @test eltype(y) == Float32
                @test norm(y - transpose(A) * x) / norm(transpose(A) * x) < tol
                y = adjoint(mat) * x
                @test eltype(y) == Float32
                @test norm(y - adjoint(A) * x) / norm(adjoint(A) * x) < tol
            end
        end
    end
end

##

using BEAST
using CompScienceMeshes
using ParallelKMeans
using H2Trees
using AdaptiveCrossApproximation

k = 2.4567799554075624 + 0.0im

circ = CompScienceMeshes.meshcircle(1.0, 0.025)
X = BEAST.lagrangecx(circ; order=2)
tree = BlockTree(KMeansTree(X.pos, 2; minvalues=100), KMeansTree(X.pos, 2; minvalues=100))

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
using LinearAlgebra

x = rand(eltype(DLop), size(DLop, 2))
norm(DLop * x - DL2_op * x) / norm(DL2_op * x)
println("HMatnorm = ", norm(DL * x - DL2 * x) / norm(DL2 * x))

@testset "H.assemble default tree (BEAST spaces)" begin
    # H.assemble's default `tree` keyword builds an H2Trees.TwoNTree straight from
    # the BEAST space (via the H2BEASTTrees extension), unlike every other test in
    # this file which passes a pre-built tree explicitly. Regression test for the
    # `testminvalues`/`trialminvalues` keyword names expected by that call.
    let
        Γhassemble = CompScienceMeshes.meshsphere(1.0, 0.3)
        Xhassemble = raviartthomas(Γhassemble)
        ophassemble = Maxwell3D.singlelayer(; wavenumber=1.0)

        Tcompressed = AdaptiveCrossApproximation.H.assemble(
            ophassemble, Xhassemble, Xhassemble; tol=1e-3, maxrank=40
        )
        Tdense = assemble(ophassemble, Xhassemble, Xhassemble)

        @test norm(Matrix(Tcompressed) - Tdense) / norm(Tdense) < 1e-2
    end
end

@testset "H.assemble KMeans tree backend (ParallelKMeans loaded)" begin
    # ParallelKMeans is loaded above (for the KMeansTree/BlockTree cell earlier in
    # this file), so H.assemble's default tree backend should now resolve to the
    # k-means clustered tree (ACAParallelKMeansTrees extension) instead of TwoNTree.
    let
        Γkmeans = CompScienceMeshes.meshsphere(1.0, 0.3)
        Xkmeans = raviartthomas(Γkmeans)
        opkmeans = Maxwell3D.singlelayer(; wavenumber=1.0)

        @test AdaptiveCrossApproximation.defaulttreebackend() isa
            AdaptiveCrossApproximation.KMeansTreeBackend

        Tcompressed = AdaptiveCrossApproximation.H.assemble(
            opkmeans, Xkmeans, Xkmeans; tol=1e-3, maxrank=40
        )
        Tdense = assemble(opkmeans, Xkmeans, Xkmeans)

        @test norm(Matrix(Tcompressed) - Tdense) / norm(Tdense) < 1e-2
    end
end

@testset "H.assemble quadstrat resolves into near/far" begin
    let
        Γq = CompScienceMeshes.meshsphere(1.0, 0.3)
        Xq = raviartthomas(Γq)
        opq = Maxwell3D.singlelayer(; wavenumber=1.0)

        qs = BEAST.defaultquadstrat(opq, Xq, Xq)
        @test AdaptiveCrossApproximation.tofarquadstrat(qs) ==
            BEAST.DoubleNumQStrat(qs.outer_rule_far, qs.inner_rule_far)

        # An explicit quadstrat override propagates to both near and far by default.
        customqs = BEAST.DoubleNumWiltonSauterQStrat(1, 1, 3, 3, 2, 2, 2, 2)
        hmat = AdaptiveCrossApproximation.H.assemble(
            opq, Xq, Xq; tol=1e-3, maxrank=40, quadstrat=customqs
        )
        @test hmat isa AdaptiveCrossApproximation.HMatrix
    end
end
