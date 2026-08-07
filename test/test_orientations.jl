using AdaptiveCrossApproximation
using LinearAlgebra
using StaticArrays
using Test

@testset "orientation_ids / normal_orientation_ids" begin
    keys = [(1, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 0), (0, 1, 0), (0, 0, 0)]

    ids, nids = AdaptiveCrossApproximation.orientation_ids(keys)
    @test nids == 3
    @test ids == [1, 2, 1, 3, 2, 3]

    nids2, ids2 = let (ids, n) = AdaptiveCrossApproximation.normal_orientation_ids(keys)
        n, ids
    end
    @test nids2 == 2
    @test ids2 == [1, 2, 1, 0, 2, 0]
end

@testset "_canonical_direction_key / _iszero_direction_key" begin
    @test AdaptiveCrossApproximation._canonical_direction_key(SVector(1.0, 0.0, 0.0)) ==
        (10, 0, 0)
    @test AdaptiveCrossApproximation._canonical_direction_key(SVector(-1.0, 0.0, 0.0)) ==
        (10, 0, 0)
    @test AdaptiveCrossApproximation._canonical_direction_key(
        SVector(0.0, 0.0, 0.0); digits=1
    ) == (0, 0, 0)
    @test AdaptiveCrossApproximation._iszero_direction_key((0, 0, 0))
    @test !AdaptiveCrossApproximation._iszero_direction_key((1, 0, 0))
end

@testset "_keyvector / _absdot" begin
    v = AdaptiveCrossApproximation._keyvector((3, 4, 0))
    @test isapprox(collect(v), [0.6, 0.8, 0.0])
    @test AdaptiveCrossApproximation._keyvector((0, 0, 0)) == (0.0, 0.0, 0.0)

    @test isapprox(
        AdaptiveCrossApproximation._absdot((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)), 1.0
    )
    @test isapprox(
        AdaptiveCrossApproximation._absdot((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), 0.0
    )
    @test isapprox(
        AdaptiveCrossApproximation._absdot((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)), 1.0
    )
end

@testset "basisfunction_orientation_ids" begin
    edges = [
        SVector(1.0, 0.0, 0.0),
        SVector(0.0, 1.0, 0.0),
        SVector(-1.0, 0.0, 0.0),
        SVector(0.0, 1.0, 0.0),
    ]
    normals = [
        SVector(0.0, 0.0, 1.0),
        SVector(0.0, 0.0, 0.0),
        SVector(0.0, 0.0, 1.0),
        SVector(0.0, 0.0, -1.0),
    ]

    edgeids, normalids, nedgeids, nnormalids = AdaptiveCrossApproximation.basisfunction_orientation_ids(
        edges, normals
    )

    @test nedgeids == 2
    @test edgeids == [1, 2, 1, 2]
    @test nnormalids == 1
    @test normalids == [1, 0, 1, 1]
end

@testset "_key_representatives" begin
    keys = [(1, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 0)]
    ids, nids = AdaptiveCrossApproximation.normal_orientation_ids(keys)
    reps = AdaptiveCrossApproximation._key_representatives(keys, ids, nids)
    @test reps[1] == (1, 0, 0)
    @test reps[2] == (0, 1, 0)
end

@testset "node_normal_orientation_sets" begin
    struct OrientTree
        nodes::Vector{Int}
        nodevalues::Vector{Vector{Int}}
        nodechildren::Vector{Vector{Int}}
        nodeparent::Vector{Int}
    end
    AdaptiveCrossApproximation.values(tree::OrientTree, node::Int) = tree.nodevalues[node]
    AdaptiveCrossApproximation.children(tree::OrientTree, node::Int) =
        tree.nodechildren[node]
    AdaptiveCrossApproximation.parent(tree::OrientTree, node::Int) = tree.nodeparent[node]

    # root (all 8), left half (1:4, normal ex), right half (5:8, normal ey)
    tree = OrientTree(
        collect(1:7),
        [collect(1:8), collect(1:4), collect(5:8), [1, 2], [3, 4], [5, 6], [7, 8]],
        [[2, 3], [4, 5], [6, 7], Int[], Int[], Int[], Int[]],
        [0, 1, 1, 2, 2, 3, 3],
    )
    normals = vcat(fill(SVector(1.0, 0.0, 0.0), 4), fill(SVector(0.0, 1.0, 0.0), 4))

    nodesets, normalids, nnormalids = AdaptiveCrossApproximation.node_normal_orientation_sets(
        normals, tree
    )

    @test nnormalids == 2
    @test normalids == [1, 1, 1, 1, 2, 2, 2, 2]
    @test nodesets[1] == Int32[1, 2]   # root sees both, orthogonal directions
    @test nodesets[2] == Int32[1]      # left half: only ex
    @test nodesets[3] == Int32[2]      # right half: only ey
    @test nodesets[4] == Int32[1]
    @test nodesets[5] == Int32[1]
    @test nodesets[6] == Int32[2]
    @test nodesets[7] == Int32[2]

    # explicit 7-arg core form should agree
    normal_keys = [AdaptiveCrossApproximation._canonical_direction_key(n) for n in normals]
    ids2, nids2 = AdaptiveCrossApproximation.normal_orientation_ids(normal_keys)
    reps2 = AdaptiveCrossApproximation._key_representatives(normal_keys, ids2, nids2)
    nodesets2 = AdaptiveCrossApproximation.node_normal_orientation_sets(
        ids2, tree, nids2, reps2
    )
    @test nodesets2 == nodesets
end

@testset "rwgorientations fallback" begin
    @test_throws ErrorException AdaptiveCrossApproximation.rwgorientations(1)
end
