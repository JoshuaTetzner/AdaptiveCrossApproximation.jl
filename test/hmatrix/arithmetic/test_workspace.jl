using AdaptiveCrossApproximation
using H2Trees
using LinearAlgebra
using StaticArrays
using Test

@testset "workspace.jl" begin
    @testset "_TreeView / _bind!" begin
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

                @testset "_TreeView / _bind!" begin
                    if symmetric
                        @test_throws ArgumentError ACA._TreeView(core)
                    else
                        view = ACA._TreeView(core)
                        @test view.near !== core.near
                        @test all(
                            view.near[i] === core.near[i] for i in eachindex(core.near)
                        )
                        @test all(view.far[i] === core.far[i] for i in eachindex(core.far))
                        other = copy(core)
                        @test other.csr === core.csr
                        @test other.tree === core.tree
                        @test all(
                            !Base.mightalias(other.near[i], core.near[i]) for
                            i in eachindex(core.near)
                        )
                        ACA._bind!(view, other)
                        @test all(
                            view.near[i] === other.near[i] for i in eachindex(core.near)
                        )
                        @test all(view.far[i] === other.far[i] for i in eachindex(core.far))
                        @test Matrix(core) ≈ reference
                        (; core) = arithmetic_workspace_fixture(ordering; symmetric)
                        @test_throws ArgumentError ACA._bind!(view, core)
                    end
                end
            end
        end
    end
end
