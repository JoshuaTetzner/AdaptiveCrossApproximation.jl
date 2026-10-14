using AdaptiveCrossApproximation
using H2Trees
using Random
using StaticArrays
using Test

function configured_isnear(η)
    return AdaptiveCrossApproximation.isnear(η; crosslevel=false)
end

# Expand traversal results into entry counts to detect omitted or repeated pairs.
function interaction_coverage(tree; spaceordering, isnear, symmetric=false)
    ACA = AdaptiveCrossApproximation
    n = H2Trees.numberofvalues(H2Trees.testtree(tree))
    near = zeros(Int, n, n)
    far = zeros(Int, n, n)
    nears = if symmetric
        ACA.symmetricnearinteractions(tree; spaceordering, isnear)
    else
        ACA.nearinteractions(tree; spaceordering, isnear)
    end
    if symmetric
        diagonals, rows, cols = nears
        for indices in diagonals
            near[indices, indices] .+= 1
        end
        for (row, col) in zip(rows, cols)
            near[row, col] .+= 1
            near[col, row] .+= 1
        end
    else
        rows, cols = nears
        for (row, col) in zip(rows, cols)
            columns = if spaceordering isa ACA.PermuteSpaceInPlace
                collect(Iterators.flatten(col))
            else
                col
            end
            near[row, columns] .+= 1
        end
    end

    fars = if symmetric
        ACA.symmetricfarinteractions(tree; spaceordering, isnear)
    else
        ACA.farinteractions(tree; spaceordering, isnear)
    end
    rows, ptr, cols = fars[1:3]
    for node in eachindex(rows), idx in ptr[node]:(ptr[node + 1] - 1)
        far[rows[node], cols[idx]] .+= 1
        if symmetric && !fars[4][idx]
            far[cols[idx], rows[node]] .+= 1
        end
    end
    return near, far
end

@testset "Interaction symmetry and space ordering" begin
    ACA = AdaptiveCrossApproximation
    rng = MersenneTwister(824)
    points = [SVector(Float64(i), 0.0) for i in randperm(rng, 48)]
    for isnear in (ACA.isnear(0.5), (a, b, i, j) -> true, (a, b, i, j) -> false)
        cluster = if isdefined(H2Trees, :TwoNTreeBuilder)
            H2Trees.buildtree(
                points; builder=H2Trees.TwoNTreeBuilder(; minhalfsize=0.1, minvalues=3)
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
        diagonalrows, rows, cols = ACA.symmetricnearinteractions(tree; isnear)
        fullrows, _ = ACA.nearinteractions(tree; isnear)
        @test allunique(rows)
        @test rows == filter(row -> row in rows, fullrows)
        @test all(allunique, cols)
        @test all(isempty(intersect(row, col)) for (row, col) in zip(rows, cols))
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
        _, rows, cols = ACA.symmetricnearinteractions(
            tree; isnear, spaceordering=ACA.PermuteSpaceInPlace()
        )
        fullrows, _ = ACA.nearinteractions(
            tree; isnear, spaceordering=ACA.PermuteSpaceInPlace()
        )
        @test allunique(rows)
        @test rows == filter(row -> row in rows, fullrows)
        @test all(allunique, cols)
    end
end

@testset "Cross-level admissibility setting" begin
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
        @test ACA.isnear(criterion; crosslevel=(!crosslevel)) isa
            ACA.IsNearFunctor{Float64,!crosslevel}
        @test ACA.isnear(criterion) === criterion
        if crosslevel
            for ordering in (ACA.PreserveSpaceOrder(), ACA.PermuteSpaceInPlace())
                for interactions in (
                    ACA.nearinteractions,
                    ACA.farinteractions,
                    ACA.symmetricnearinteractions,
                    ACA.symmetricfarinteractions,
                )
                    @test_throws ArgumentError interactions(
                        tree; isnear=criterion, spaceordering=ordering
                    )
                end
            end
        end
    end
end
