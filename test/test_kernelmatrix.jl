using AdaptiveCrossApproximation
using BEAST
using CompScienceMeshes
using Test

@testset "KernelMatrix" begin
    struct EmptyKernelMatrix <: AdaptiveCrossApproximation.AbstractKernelMatrix{Float64} end

    Γ = meshicosphere(2, 1.0)
    x = lagrangec0d1(Γ)
    y = lagrangec0d1(Γ)
    op = Helmholtz3D.singlelayer()

    A = AdaptiveCrossApproximation.AbstractKernelMatrix(op, x, y)
    P = AdaptiveCrossApproximation.AbstractKernelMatrix(
        (x, y) -> sum(x + y), Γ.vertices, Γ.vertices
    )

    @test size(A) == size(P) == (length(x), length(y))
    @test eltype(A) == eltype(P) == scalartype(op) == Float64
    @test size(P, 1) == length(x)
    @test size(P, 2) == length(y)
    @test_throws ErrorException size(P, 3)
    @test_throws ErrorException AdaptiveCrossApproximation.AbstractKernelMatrix(
        nothing, nothing, nothing
    )
    emptykernelmatrix = EmptyKernelMatrix()
    @test eltype(emptykernelmatrix) == Float64
    @test_throws ArgumentError emptykernelmatrix(nothing, nothing, nothing)

    rows = [[1, 3], [2]]
    columns = [[2, 4], [1, 3]]
    blocks = [zeros(length(rows[i]), length(columns[i])) for i in eachindex(rows)]
    returnedblocks = AdaptiveCrossApproximation.assemble_blocks(P, blocks, rows, columns)
    @test returnedblocks === blocks
    for i in eachindex(blocks)
        expectedblock = [
            sum(Γ.vertices[m] + Γ.vertices[n]) for m in rows[i], n in columns[i]
        ]
        @test blocks[i] == expectedblock
    end

    block = zeros(2, 2)
    AdaptiveCrossApproximation.nextrc!(block, P, rows[1], columns[1])
    @test block == blocks[1]
    @test AdaptiveCrossApproximation.localkernelmatrix(P, rows[1], columns[1]) === P

    Γ = meshicosphere(2, Float32(1.0))
    x = lagrangec0d1(Γ)
    y = lagrangec0d1(Γ)
    op = Helmholtz3D.singlelayer(; gamma=Float32(1.0))

    A = AdaptiveCrossApproximation.AbstractKernelMatrix(op, x, y)
    P = AdaptiveCrossApproximation.AbstractKernelMatrix(
        (x, y) -> sum(x + y), Γ.vertices, Γ.vertices
    )
    @test size(A) == size(P) == (length(x), length(y))
    @test eltype(A) == eltype(P) == scalartype(op) == Float32
end
