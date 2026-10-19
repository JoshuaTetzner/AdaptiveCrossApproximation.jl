using AdaptiveCrossApproximation
using H2Trees
using LinearAlgebra
using Random
using StaticArrays
using Test
using BEAST
using CompScienceMeshes

@testset "abstracthmatrix.jl" begin
    @testset "PermuteSpaceInPlace" begin
        points = [SVector(rand(), rand(), 0.0) for _ in 1:20]
        tree = H2Trees.buildtree(
            points,
            points;
            builder=H2Trees.BlockTreeBuilder(;
                test=H2Trees.TwoNTreeBuilder(; minhalfsize=1 / 2^10, minvalues=10),
                trial=H2Trees.TwoNTreeBuilder(; minhalfsize=1 / 2^10, minvalues=10),
            ),
        )
        sharedtree = H2Trees.testtree(tree)
        tree = H2Trees.BlockTree(sharedtree, sharedtree)

        testspace, trialspace = AdaptiveCrossApproximation.PermuteSpaceInPlace()(
            tree, points, points
        )
        @test testspace === trialspace && trialspace === points

        mat = AdaptiveCrossApproximation.HMatrix(
            fct,
            points,
            tree;
            spaceordering=AdaptiveCrossApproximation.PermuteSpaceInPlace(),
            tol=1e-4,
            verbose=false,
        )
        index = AdaptiveCrossApproximation._TreeIndex(mat)
        @test index.rowperm isa AdaptiveCrossApproximation.IdentityPermutation
        @test index.colperm isa AdaptiveCrossApproximation.IdentityPermutation
    end
end
