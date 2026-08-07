using AdaptiveCrossApproximation
using LinearAlgebra
using LinearMaps
using Random
using Test

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

@testset "utils helpers" begin
    # collectassigned only makes sense for a reference (non-bits) element type:
    # bitstype vectors (e.g. Vector{Int}) are always fully "assigned" once allocated.
    v = Vector{Vector{Int}}(undef, 5)
    v[2] = [20]
    v[4] = [40]
    compact = AdaptiveCrossApproximation.collectassigned(v)
    @test compact == [[20], [40]]

    vals = [1, 3, 4]
    nearvals = [[10, 11], [20], [30, 31, 32]]
    assigned = [true, false, true]
    cv, cn = AdaptiveCrossApproximation.collectnears(vals, nearvals, assigned)
    @test cv == [1, 4]
    @test cn == [[10, 11], [30, 31, 32]]

    ptr, data = AdaptiveCrossApproximation.linearizestorage([[1, 2], Int[], [3]])
    @test ptr == [1, 3, 3, 4]
    @test data == [1, 2, 3]

    valsb = [[1, 2, 3], [4, 5]]
    ptrb = [1, 2, 3]
    farvalsb = [[10, 20], [30]]
    blen, fblen = AdaptiveCrossApproximation.buffersize(valsb, ptrb, farvalsb, [1, 2])
    @test blen == 3
    @test fblen == 2

    levelvals, levelfarvals, levelidcs = AdaptiveCrossApproximation.blockvalues(
        valsb, ptrb, farvalsb, [1, 2]
    )
    @test levelvals == [[1, 2, 3], [4, 5]]
    @test levelfarvals == [[10, 20], [30]]
    @test levelidcs == [1, 2]
end
