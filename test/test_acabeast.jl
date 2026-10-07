using AdaptiveCrossApproximation
using BEAST
using CompScienceMeshes
using LinearAlgebra
using Random
using Test
Random.seed!(1234)

λ = 10
k = 2 * π / λ

# 3D tests
@testset "ACA BEAST 3D" begin
    plate = meshrectangle(1.0, 1.0, 0.2)
    facingplates = weld(plate, translate(plate, [0.0, 0.0, 1.0]))

    meshes = [
        (plate, translate(plate, [0.0, 0.0, 3.0])),
        (facingplates, translate(facingplates, [3.0, 0.0, 0.0])),
        (facingplates, translate(plate, [3.0, 0.0, 0.0])),
        (plate, translate(facingplates, [3.0, 0.0, 0.0])),
    ]

    for mesh in meshes
        Xs = raviartthomas(mesh[2])
        Xt = raviartthomas(mesh[1])
        Yt = buffachristiansen(mesh[1])
        X1 = lagrangec0d1(mesh[2])
        Y1 = duallagrangec0d1(mesh[1])

        Os = [
            (Helmholtz3D.singlelayer(;), Y1, X1)
            (Helmholtz3D.hypersingular(;), Y1, X1)
            (Maxwell3D.singlelayer(; wavenumber=k), Xt, Xs)
        ]
        DLOs = [
            (Helmholtz3D.doublelayer(;), Y1, X1)
            (Helmholtz3D.doublelayer_transposed(;), Y1, X1)
            (Maxwell3D.doublelayer(; wavenumber=k), Yt, Xs)
        ]
        quadstrat = BEAST.DoubleNumQStrat(1, 2)
        for (O, Y, X) in Os
            A = assemble(O, Y, X; quadstrat=quadstrat)

            for tol in [1e-3, 1e-4, 1e-6]
                local comp = AdaptiveCrossApproximation.defaultcompressor(O, Y, X; tol=tol)
                @test comp.rowpivoting isa AdaptiveCrossApproximation.MaximumValue
                @test comp.columnpivoting isa AdaptiveCrossApproximation.MaximumValue
                @test comp.convergence isa AdaptiveCrossApproximation.FNormEstimator
                @test comp.convergence.tol == tol

                local rowbuffer = zeros(scalartype(O), length(Y), length(X))
                local colbuffer = zeros(scalartype(O), length(Y), length(X))
                local K = AdaptiveCrossApproximation.AbstractKernelMatrix(
                    O, Y, X; matrixdata=quadstrat
                )

                local npivots = comp(K, colbuffer, rowbuffer, min(length(Y), length(X)))
                @test norm(A - colbuffer[:, 1:npivots] * rowbuffer[1:npivots, :]) /
                      norm(A) < 5tol
            end
        end

        for (O, Y, X) in DLOs
            A = assemble(O, Y, X; quadstrat=quadstrat)
            for tol in [1e-3, 1e-4, 1e-6]
                local comp = AdaptiveCrossApproximation.defaultcompressor(O, Y, X; tol=tol)
                @test comp.rowpivoting isa AdaptiveCrossApproximation.CombinedPivStrat
                @test comp.columnpivoting isa AdaptiveCrossApproximation.MaximumValue
                @test comp.convergence isa AdaptiveCrossApproximation.CombinedConvCrit
                @test comp.convergence.crits[1] isa
                    AdaptiveCrossApproximation.FNormEstimator
                @test comp.convergence.crits[1].tol == tol
                @test comp.convergence.crits[2] isa
                    AdaptiveCrossApproximation.RandomSampling
                @test comp.convergence.crits[2].tol == tol

                local rowbuffer = zeros(scalartype(O), length(Y), length(X))
                local colbuffer = zeros(scalartype(O), length(Y), length(X))
                local K = AdaptiveCrossApproximation.AbstractKernelMatrix(
                    O, Y, X; matrixdata=quadstrat
                )
                local npivots = comp(K, colbuffer, rowbuffer, min(length(Y), length(X)))
                @test norm(A - colbuffer[:, 1:npivots] * rowbuffer[1:npivots, :]) /
                      norm(A) < 2tol
            end
        end
    end
end

@testset "Local BEAST KernelMatrix" begin
    mesh = meshicosphere(2, 1.0)
    space = raviartthomas(mesh)
    operator = Maxwell3D.singlelayer(; wavenumber=k)
    quadstrat = BEAST.DoubleNumQStrat(1, 2)
    matrix = AdaptiveCrossApproximation.AbstractKernelMatrix(
        operator, space, space; matrixdata=quadstrat
    )
    rows = collect(1:3)
    columns = collect(4:8)
    localmatrix = AdaptiveCrossApproximation.localkernelmatrix(matrix, rows, columns)
    extension_module = Base.get_extension(AdaptiveCrossApproximation, :ACABEAST)
    fallbackmatrix = AdaptiveCrossApproximation.AbstractKernelMatrix(operator, space, space)

    @test extension_module !== nothing
    @test localmatrix isa extension_module.LocalBEASTKernelMatrix
    @test AdaptiveCrossApproximation.localkernelmatrix(fallbackmatrix, rows, columns) ===
        fallbackmatrix
    @test size(localmatrix) == size(matrix)
    @test size(localmatrix, 1) == size(matrix, 1)
    @test size(localmatrix, 2) == size(matrix, 2)
    @test eltype(localmatrix) == eltype(matrix)

    fullblock = zeros(ComplexF64, length(rows), length(columns))
    matrix(fullblock, rows, columns)

    block = zeros(ComplexF64, length(rows), length(columns))
    localmatrix(block, rows, columns)
    @test block ≈ fullblock

    rowblock = zeros(ComplexF64, 1, length(columns))
    localmatrix(rowblock, rows[1], columns)
    @test rowblock ≈ fullblock[1:1, :]

    columnblock = zeros(ComplexF64, length(rows), 1)
    localmatrix(columnblock, rows, columns[1])
    @test columnblock ≈ fullblock[:, 1:1]

    scalarblock = zeros(ComplexF64, 1, 1)
    localmatrix(scalarblock, rows[1], columns[1])
    @test scalarblock[1] ≈ fullblock[1, 1]

    fill!(rowblock, 0)
    AdaptiveCrossApproximation.nextrc!(rowblock, localmatrix, rows[1], columns)
    @test rowblock ≈ fullblock[1:1, :]

    fill!(columnblock, 0)
    AdaptiveCrossApproximation.nextrc!(columnblock, localmatrix, rows, columns[1])
    @test columnblock ≈ fullblock[:, 1:1]

    fill!(rowblock, 0)
    AdaptiveCrossApproximation.nextrc!(rowblock, matrix, rows[1], columns)
    @test rowblock ≈ fullblock[1:1, :]

    fill!(block, 0)
    AdaptiveCrossApproximation.nextrc!(block, matrix, rows, columns)
    @test block ≈ fullblock

    fill!(columnblock, 0)
    AdaptiveCrossApproximation.nextrc!(columnblock, matrix, rows, columns[1])
    @test columnblock ≈ fullblock[:, 1:1]

    fill!(scalarblock, 0)
    AdaptiveCrossApproximation.nextrc!(scalarblock, matrix, rows[1], columns[1])
    @test scalarblock[1] ≈ fullblock[1, 1]
end

@testset "ACA BEAST 2D" begin
    line = meshsegment(1.0, 0.05)
    facinglines = weld(line, translate(line, [0.0, 1.0]))
    meshes = [
        (line, translate(line, [0.0, 1.0])),
        (facinglines, translate(facinglines, [1.0, 0.0])),
        (facinglines, translate(line, [3.0, 0.0])),
        (line, translate(facinglines, [3.0, 0.0])),
    ]
    quadstrat = BEAST.DoubleNumQStrat(1, 2)
    for mesh in meshes
        X1 = lagrangec0d1(mesh[2])
        Y1 = lagrangec0d1(mesh[1])

        Os = [
            (Helmholtz2D.singlelayer(; wavenumber=k), Y1, X1)
            (Helmholtz2D.hypersingular(; wavenumber=k), Y1, X1)
        ]
        DLOs = [(Helmholtz2D.doublelayer(; wavenumber=k), Y1, X1)]

        for (O, Y, X) in Os
            A = assemble(O, Y, X; quadstrat=quadstrat)

            for tol in [1e-3, 1e-4, 1e-6]
                local comp = AdaptiveCrossApproximation.defaultcompressor(O, Y, X; tol=tol)
                @test comp.rowpivoting isa AdaptiveCrossApproximation.MaximumValue
                @test comp.columnpivoting isa AdaptiveCrossApproximation.MaximumValue
                @test comp.convergence isa AdaptiveCrossApproximation.FNormEstimator
                @test comp.convergence.tol == tol

                local rowbuffer = zeros(scalartype(O), length(Y), length(X))
                local colbuffer = zeros(scalartype(O), length(Y), length(X))
                local K = AdaptiveCrossApproximation.AbstractKernelMatrix(
                    O, Y, X; matrixdata=quadstrat
                )

                local npivots = comp(K, colbuffer, rowbuffer, min(length(Y), length(X)))
                @test norm(A - colbuffer[:, 1:npivots] * rowbuffer[1:npivots, :]) /
                      norm(A) < 2tol
            end
        end

        for (O, Y, X) in DLOs
            A = assemble(O, Y, X; quadstrat=quadstrat)
            for tol in [1e-3, 1e-4, 1e-6]
                local comp = AdaptiveCrossApproximation.defaultcompressor(O, Y, X; tol=tol)
                @test comp.rowpivoting isa AdaptiveCrossApproximation.CombinedPivStrat
                @test comp.columnpivoting isa AdaptiveCrossApproximation.MaximumValue
                @test comp.convergence isa AdaptiveCrossApproximation.CombinedConvCrit
                @test comp.convergence.crits[1] isa
                    AdaptiveCrossApproximation.FNormEstimator
                @test comp.convergence.crits[1].tol == tol
                @test comp.convergence.crits[2] isa
                    AdaptiveCrossApproximation.RandomSampling
                @test comp.convergence.crits[2].tol == tol

                local rowbuffer = zeros(scalartype(O), length(Y), length(X))
                local colbuffer = zeros(scalartype(O), length(Y), length(X))
                local K = AdaptiveCrossApproximation.AbstractKernelMatrix(
                    O, Y, X; matrixdata=quadstrat
                )
                Random.seed!(1234)
                local npivots = comp(K, colbuffer, rowbuffer, min(length(Y), length(X)))
                @test norm(A - colbuffer[:, 1:npivots] * rowbuffer[1:npivots, :]) /
                      norm(A) < 2tol
            end
        end
    end
end
