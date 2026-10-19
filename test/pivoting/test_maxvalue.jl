using AdaptiveCrossApproximation
using LinearAlgebra
using Random
using StaticArrays
using Test

@testset "maxvalue.jl" begin
    @testset "MaximumValue" begin
        piv = AdaptiveCrossApproximation.MaximumValue()
        idcs = [10, 20, 30, 40]
        functor = piv(idcs)

        @test functor() == 1
        next = functor([0.2, -3.0, 1.0, 2.5])
        @test next == 2

        AdaptiveCrossApproximation.reset!(functor, [1, 2, 3])
        @test length(functor.usedidcs) >= 3
        @test all(.!view(functor.usedidcs, 1:3))

        resize!(functor, 6)
        @test length(functor.usedidcs) == 6

        AdaptiveCrossApproximation.reset!(functor, [9, 8, 7, 6, 5, 4])
        @test all(.!view(functor.usedidcs, 1:6))

        resize!(functor, 9)
        AdaptiveCrossApproximation.reset!(functor, [1, 2, 3])
        @test length(functor.usedidcs) == 9
        @test functor.nactive == 3
        @test functor([0.1, -2.0, 1.5]) == 2
    end
end
