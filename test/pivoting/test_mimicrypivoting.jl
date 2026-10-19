using AdaptiveCrossApproximation
using LinearAlgebra
using Random
using StaticArrays
using Test

@testset "mimicrypivoting.jl" begin
    @testset "MimicryPivoting" begin
        refpos = [@SVector [Float64(i), 0.0, 0.0] for i in 1:6]
        pos = [@SVector [Float64(i), 1.0, 0.0] for i in 1:8]
        piv = AdaptiveCrossApproximation.MimicryPivoting(refpos, pos)
        functor = piv([1, 2, 3], [2, 4, 6, 8])

        idbuf = functor.idcs
        hbuf = functor.h
        lejabuf = functor.leja
        wbuf = functor.w

        first = functor()

        reset!(functor, [2, 3], [1, 3, 5])
        @test functor.idcs === idbuf
        @test functor.h === hbuf
        @test functor.leja === lejabuf
        @test functor.w === wbuf
        @test functor.nactive == 3
        @test collect(view(functor.idcs, 1:3)) == [1, 3, 5]

        resize!(functor, 10)
        @test length(functor.idcs) == 10
        @test length(functor.h) == 10
        @test length(functor.leja) == 10
        @test length(functor.w) == 10
        @test functor.nactive == 10

        reset!(functor, [1, 4, 6], [2, 5, 8, 7])
        @test functor.nactive == 4
        @test collect(view(functor.idcs, 1:4)) == [2, 5, 8, 7]
        @test all(iszero, view(functor.h, 1:4))
        @test all(isone, view(functor.leja, 1:4))
    end
end
