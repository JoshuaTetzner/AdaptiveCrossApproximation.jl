using AdaptiveCrossApproximation
using LinearAlgebra
using Random
using StaticArrays
using Test

@testset "efiedirectional.jl" begin
    @testset "EFIEDirectionalFilter" begin
        struct DirTree
            nodes::Vector{Int}
            centers::Vector{SVector{3,Float64}}
            nodevalues::Vector{Vector{Int}}
            nodechildren::Vector{Vector{Int}}
            nodeparent::Vector{Int}
        end

        AdaptiveCrossApproximation.center(tree::DirTree, node::Int) = tree.centers[node]
        AdaptiveCrossApproximation.values(tree::DirTree, node::Int) = tree.nodevalues[node]
        AdaptiveCrossApproximation.values(tree::DirTree, nodes::Vector{Int}) =
            reduce(vcat, (tree.nodevalues[n] for n in nodes); init=Int[])
        AdaptiveCrossApproximation.children(tree::DirTree, node::Int) =
            tree.nodechildren[node]
        AdaptiveCrossApproximation.parent(tree::DirTree, node::Int) = tree.nodeparent[node]
        function AdaptiveCrossApproximation.firstchild(tree::DirTree, node::Int)
            isempty(tree.nodechildren[node]) && return 0
            return first(tree.nodechildren[node])
        end

        # left half (nodes 2,4,5 / basis 1:4) carries direction 1 (ex), right half
        # (nodes 3,6,7 / basis 5:8) carries direction 2 (ey); root sees both.
        tree = DirTree(
            collect(1:7),
            [
                SVector(4.5, 0.0, 0.0),
                SVector(2.5, 0.0, 0.0),
                SVector(6.5, 0.0, 0.0),
                SVector(1.5, 0.0, 0.0),
                SVector(3.5, 0.0, 0.0),
                SVector(5.5, 0.0, 0.0),
                SVector(7.5, 0.0, 0.0),
            ],
            [collect(1:8), collect(1:4), collect(5:8), [1, 2], [3, 4], [5, 6], [7, 8]],
            [[2, 3], [4, 5], [6, 7], Int[], Int[], Int[], Int[]],
            [0, 1, 1, 2, 2, 3, 3],
        )

        edges = fill(SVector(1.0, 0.0, 0.0), 8)
        normals = vcat(fill(SVector(1.0, 0.0, 0.0), 4), fill(SVector(0.0, 1.0, 0.0), 4))
        edgeids, normalids, nedgeids, nnormalids = AdaptiveCrossApproximation.basisfunction_orientation_ids(
            edges, normals
        )
        nodesets, trialnormalids, ntrialnormalids = AdaptiveCrossApproximation.node_normal_orientation_sets(
            normals, tree
        )
        @test normalids == trialnormalids

        refpos = [@SVector [Float64(i), 0.0, 0.0] for i in 1:8]
        pos = [@SVector [Float64(i), 1.0, 0.0] for i in 1:8]

        filter = AdaptiveCrossApproximation.EFIEDirectionalFilter(
            edgeids, normalids, nodesets
        )
        piv = AdaptiveCrossApproximation.TreeMimicryPivoting(refpos, pos, tree, filter)
        maxrank = 5
        convcrit = AdaptiveCrossApproximation.PhaseExtrapolator(1e-6)(maxrank)
        functor = piv(convcrit, [1, 8], [2, 3], maxrank)

        # with lastconverged never set (no residual data fed back), the direction
        # filter round-robins between the two available directions every pivot
        used = Int[functor()]
        dirs = Int32[normalids[used[1]]]
        for n in 2:4
            p = functor(n)
            push!(used, p)
            push!(dirs, normalids[p])
        end

        @test length(unique(used)) == 4
        @test dirs == Int32[1, 2, 1, 2]
        @test all(in(1:4), used[1:2:end])
        @test all(in(5:8), used[2:2:end])
    end
end
