using AdaptiveCrossApproximation
using LinearAlgebra
using Random
using StaticArrays
using Test

@testset "treemimicrypivoting.jl" begin
    @testset "TreeMimicryPivoting" begin
        struct MockTree
            centers::Vector{SVector{3,Float64}}
            nodevalues::Vector{Vector{Int}}
            nodechildren::Vector{Vector{Int}}
            nodeparent::Vector{Int}
        end

        AdaptiveCrossApproximation.center(tree::MockTree, node::Int) = tree.centers[node]
        AdaptiveCrossApproximation.values(tree::MockTree, node::Int) = tree.nodevalues[node]
        AdaptiveCrossApproximation.values(tree::MockTree, nodes::Vector{Int}) =
            reduce(vcat, (tree.nodevalues[n] for n in nodes); init=Int[])
        AdaptiveCrossApproximation.children(tree::MockTree, node::Int) =
            tree.nodechildren[node]
        AdaptiveCrossApproximation.parent(tree::MockTree, node::Int) = tree.nodeparent[node]
        function AdaptiveCrossApproximation.firstchild(tree::MockTree, node::Int)
            if isempty(tree.nodechildren[node])
                return 0
            else
                return first(tree.nodechildren[node])
            end
        end

        tree = MockTree(
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

        refpos = [@SVector [Float64(i), 0.0, 0.0] for i in 1:8]
        pos = [@SVector [Float64(i), 1.0, 0.0] for i in 1:8]
        piv = AdaptiveCrossApproximation.TreeMimicryPivoting(refpos, pos, tree)
        functor = piv([1, 8], [2, 3], 5)

        @test length(functor.usedidcs) == 5
        @test length(functor.emptyclusters) == 5
        @test functor.nactive == 2
        @test length(functor.h) == 2
        @test length(functor.leja) == 2
        @test length(functor.w) == 2
        @test collect(view(functor.farfield, 1:2)) == [2, 3]

        firstpivot = functor()
        @test firstpivot in 1:8
        secondpivot = functor(2)
        @test secondpivot in 1:8

        usedbuf = functor.usedidcs
        emptybuf = functor.emptyclusters
        farfieldbuf = functor.farfield
        hbuf = functor.h
        lejabuf = functor.leja
        wbuf = functor.w

        resize!(functor, 9)
        @test length(functor.farfield) == 9
        @test length(functor.h) == 9
        @test length(functor.leja) == 9
        @test length(functor.w) == 9
        @test length(functor.usedidcs) == 5
        @test length(functor.emptyclusters) == 5
        @test functor.nactive == 9

        resize!(functor, 4)
        @test length(functor.farfield) == 9
        @test length(functor.h) == 9
        @test length(functor.leja) == 9
        @test length(functor.w) == 9
        @test functor.nactive == 4

        reset!(functor, [1, 8], [2, 3])
        @test functor.usedidcs === usedbuf
        @test functor.emptyclusters === emptybuf
        @test functor.farfield === farfieldbuf
        @test functor.h === hbuf
        @test functor.leja === lejabuf
        @test functor.w === wbuf
        @test functor.nempty == 0
        @test functor.nactive == 2
        @test collect(view(functor.farfield, 1:2)) == [2, 3]
        @test all(iszero, view(functor.h, 1:2))
        @test all(isone, view(functor.leja, 1:2))
        @test all(iszero, view(functor.w, 1:2))
        @test all(iszero, view(functor.usedidcs, 1:5))
        @test all(iszero, view(functor.emptyclusters, 1:5))

        reset!(functor, [1, 8], [2, 3, 4])
        @test functor.nactive == 3
        @test collect(view(functor.farfield, 1:3)) == [2, 3, 4]
        @test length(functor.farfield) == 9
        @test length(functor.h) == 9
        @test length(functor.leja) == 9
        @test length(functor.w) == 9
        @test length(functor.usedidcs) == 5
        @test length(functor.emptyclusters) == 5
    end
end
