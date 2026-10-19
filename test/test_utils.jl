using AdaptiveCrossApproximation
using LinearAlgebra
using LinearMaps
using Random
using Test

@testset "utils.jl" begin
    @testset "LowRankMatrix" begin
        Random.seed!(42)
        U = rand(5, 3)
        V = rand(3, 4)
        lrm = AdaptiveCrossApproximation.LowRankMatrix(U, V)

        @test size(lrm) == (5, 4)
        @test eltype(lrm) == Float64

        x = rand(4)
        @test isapprox(lrm * x, U * V * x)

        y = rand(5)
        @test isapprox(transpose(lrm) * y, transpose(U * V) * y)
        @test isapprox(adjoint(lrm) * y, adjoint(U * V) * y)

        Uc = rand(ComplexF64, 5, 3)
        Vc = rand(ComplexF64, 3, 4)
        lrmc = AdaptiveCrossApproximation.LowRankMatrix(Uc, Vc)
        xc = rand(ComplexF64, 4)
        @test isapprox(lrmc * xc, Uc * Vc * xc)
        yc = rand(ComplexF64, 5)
        @test isapprox(transpose(lrmc) * yc, transpose(Uc * Vc) * yc)
        @test isapprox(adjoint(lrmc) * yc, adjoint(Uc * Vc) * yc)

        @test_throws AssertionError AdaptiveCrossApproximation.LowRankMatrix(
            rand(5, 3), rand(4, 4)
        )
    end

    @testset "linearizestorage" begin
        ptr, nodes = AdaptiveCrossApproximation.linearizestorage([[5, 6], Int[], [7]])
        @test ptr == [1, 3, 3, 4]
        @test nodes == [5, 6, 7]
        @test AdaptiveCrossApproximation.linearizestorage(Vector{Int}[]) == ([1], Int[])
    end
end
