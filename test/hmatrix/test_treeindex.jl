using AdaptiveCrossApproximation
using H2Trees
using LinearAlgebra
using StaticArrays
using Test

@testset "treeindex.jl" begin
    @testset "_TreeIndex" begin
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

                @testset "values / _clusterranges" begin
                    @test sort(rowperm) == collect(1:size(core, 1))
                    @test sort(colperm) == collect(1:size(core, 2))
                    if ordering isa ACA.PermuteSpaceInPlace
                        @test index.rowperm isa ACA.IdentityPermutation
                        @test index.colperm isa ACA.IdentityPermutation
                        @test rowperm == collect(1:size(core, 1))
                        @test colperm == collect(1:size(core, 2))
                    else
                        @test index.rowperm == rowperm
                        @test index.colperm == colperm
                        @test rowperm != collect(1:size(core, 1))
                        @test colperm != collect(1:size(core, 2))
                    end
                    if symmetric
                        @test index.rowperm === index.colperm
                    else
                        @test rowtree !== coltree
                        @test size(core, 1) != size(core, 2)
                        @test rowperm != colperm
                    end
                    @test leavesbefore == (
                        [H2Trees.values(rowtree, n) for n in H2Trees.leaves(rowtree)],
                        [H2Trees.values(coltree, n) for n in H2Trees.leaves(coltree)],
                    )
                    rowranges = workspace_test_ranges(rowtree, rowperm)
                    colranges = workspace_test_ranges(coltree, colperm)
                    @test index.rowranges == rowranges
                    @test index.colranges == colranges
                    expected = reference[rowperm, colperm]
                    @test workspace_test_dense(core, rowranges, colranges) ≈ expected
                    @test workspace_test_dense(core, rowranges, colranges; traversal=:csc) ≈
                        expected
                    @test Matrix(core) ≈ reference
                end
            end
        end
    end
end
