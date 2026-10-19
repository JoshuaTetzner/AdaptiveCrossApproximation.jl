using AdaptiveCrossApproximation
using LinearAlgebra
using Random
using StaticArrays
using Test

@testset "lejapoints.jl" begin
    @testset "Leja2" begin
        pts = [@SVector [Float64(i), 0.0, 0.0] for i in 1:6]
        piv = AdaptiveCrossApproximation.Leja2(pts)
        idcs = [2, 4, 6]
        functor = piv(idcs)

        @test functor() == 1
        @test all(isapprox.(functor.h, [0.0, 2.0, 4.0]))

        next = functor(zeros(length(idcs)))
        @test next == 3
        @test all(isapprox.(functor.h, [0.0, 2.0, 0.0]))

        resize!(functor, 5)
        @test length(functor.h) == 5
        @test length(functor.idcs) == 5

        reset!(functor, [1, 3, 5, 6, 2])
        @test collect(view(functor.idcs, 1:5)) == [1, 3, 5, 6, 2]
        @test all(iszero, view(functor.h, 1:5))

        resize!(functor, 8)
        reset!(functor, [1, 3, 5])
        @test length(functor.h) == 8
        @test length(functor.idcs) == 8
        @test functor.nactive == 3
        @test functor(zeros(3)) == 1
        @test all(isapprox.(view(functor.h, 1:3), [0.0, 2.0, 4.0]))
    end
end
