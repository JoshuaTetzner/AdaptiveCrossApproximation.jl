using AdaptiveCrossApproximation
using Test

@testset "blockview.jl" begin
    @testset "blockview" begin
        ACA = AdaptiveCrossApproximation
        C = ACA._TreeView(arithmetic_workspace_fixture(ACA.PreserveSpaceOrder()).core)
        @test_throws BoundsError ACA.blockview(C, 0, 1)
        @test_throws BoundsError ACA.blockview(C, 0:3, 1:2)
    end
end
