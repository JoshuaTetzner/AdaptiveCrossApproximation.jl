using AdaptiveCrossApproximation
using H2Trees
using Random
using StaticArrays
using Test

function configured_isnear(η)
    return AdaptiveCrossApproximation.isnear(η; crosslevel=false)
end

# Expand the node pairs of the terminal blocks into entry counts to detect
# omitted or repeated pairs.
function interaction_coverage(tree; spaceordering, isnear, symmetric=false)
    ACA = AdaptiveCrossApproximation
    cluster = H2Trees.testtree(tree)
    n = H2Trees.numberofvalues(cluster)
    indices(node) = ACA.values(cluster, node)
    near = zeros(Int, n, n)
    far = zeros(Int, n, n)
    partition = if symmetric
        ACA.symmetricinteractions(tree; isnear)
    else
        ACA.interactions(tree; isnear)
    end
    @test length(partition.nearptr) ==
        length(partition.farptr) ==
        H2Trees.numberofnodes(cluster) + 1
    for (counts, ptr, nodes) in (
        (near, partition.nearptr, partition.nearnodes),
        (far, partition.farptr, partition.farnodes),
    )
        for a in 1:(length(ptr) - 1), k in ptr[a]:(ptr[a + 1] - 1)
            b = nodes[k]
            counts[indices(a), indices(b)] .+= 1
            symmetric && a != b && (counts[indices(b), indices(a)] .+= 1)
        end
    end
    return near, far
end

@testset "interactions.jl" begin
    @testset "interactions / symmetricinteractions" begin
        ACA = AdaptiveCrossApproximation
        rng = MersenneTwister(824)
        points = [SVector(Float64(i), 0.0) for i in randperm(rng, 48)]
        for isnear in (ACA.isnear(0.5), (a, b, i, j) -> true, (a, b, i, j) -> false)
            cluster = if isdefined(H2Trees, :TwoNTreeBuilder)
                H2Trees.buildtree(
                    points;
                    builder=H2Trees.TwoNTreeBuilder(; minhalfsize=0.1, minvalues=3),
                )
            else
                H2Trees.TwoNTree(points, 0.1; minvalues=3)
            end
            tree = H2Trees.BlockTree(cluster, cluster)
            near, far = interaction_coverage(
                tree; spaceordering=ACA.PreserveSpaceOrder(), isnear
            )
            @test all(==(1), near + far)
            @test interaction_coverage(
                tree; spaceordering=ACA.PreserveSpaceOrder(), isnear, symmetric=true
            ) == (near, far)
            symmetric = ACA.symmetricinteractions(tree; isnear)
            full = ACA.interactions(tree; isnear)
            for (ptr, nodes) in (
                (symmetric.nearptr, symmetric.nearnodes),
                (symmetric.farptr, symmetric.farnodes),
            )
                rows = 1:(length(ptr) - 1)
                @test all(a <= nodes[k] for a in rows for k in ptr[a]:(ptr[a + 1] - 1))
                @test all(allunique(view(nodes, ptr[a]:(ptr[a + 1] - 1))) for a in rows)
                # The stored pairs of a test node are a subset of its full partition.
                @test all(
                    nodes[k] in
                    view(full.nearnodes, full.nearptr[a]:(full.nearptr[a + 1] - 1)) ||
                        nodes[k] in
                        view(full.farnodes, full.farptr[a]:(full.farptr[a + 1] - 1)) for
                    a in rows for k in ptr[a]:(ptr[a + 1] - 1)
                )
            end
            if isnear isa ACA.IsNearFunctor
                @test any(!iszero, near)
                @test any(!iszero, far)
            end

            perm = reduce(
                vcat, [H2Trees.values(cluster, leaf) for leaf in H2Trees.leaves(cluster)]
            )
            ACA.PermuteSpaceInPlace()(tree, copy(points))
            expected = (near[perm, perm], far[perm, perm])
            for symmetric in (false, true)
                @test interaction_coverage(
                    tree; spaceordering=ACA.PermuteSpaceInPlace(), isnear, symmetric
                ) == expected
            end
        end
    end

    @testset "isnear (cross-level validation)" begin
        ACA = AdaptiveCrossApproximation
        points = [SVector(Float64(i), 0.0) for i in 1:12]
        cluster = if isdefined(H2Trees, :TwoNTreeBuilder)
            H2Trees.buildtree(
                points; builder=H2Trees.TwoNTreeBuilder(; minhalfsize=0.1, minvalues=3)
            )
        else
            H2Trees.TwoNTree(points, 0.1; minvalues=3)
        end
        tree = H2Trees.BlockTree(cluster, cluster)
        root = H2Trees.root(cluster)
        reference = ACA.isnear(0.5)(cluster, cluster, root, root)
        @test (@inferred configured_isnear(0.5)) isa ACA.IsNearFunctor{Float64,false}
        for crosslevel in (false, true)
            criterion = ACA.isnear(0.5; crosslevel)
            @test criterion isa ACA.IsNearFunctor{Float64,crosslevel}
            @test criterion(cluster, cluster, root, root) == reference
            if crosslevel
                for interactions in (ACA.interactions, ACA.symmetricinteractions)
                    @test_throws ArgumentError interactions(tree; isnear=criterion)
                end
            end
        end
    end
end
