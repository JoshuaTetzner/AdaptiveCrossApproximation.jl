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

    # One-sided local matrices: only the given side is prepared, the other falls back.
    rowsonly = AdaptiveCrossApproximation.localkernelmatrix(matrix, rows, nothing)
    columnsonly = AdaptiveCrossApproximation.localkernelmatrix(matrix, nothing, columns)
    @test rowsonly isa extension_module.LocalBEASTKernelMatrix
    @test columnsonly isa extension_module.LocalBEASTKernelMatrix
    @test rowsonly.rowdata === nothing
    @test columnsonly.columndata === nothing
    @test AdaptiveCrossApproximation.localkernelmatrix(fallbackmatrix, rows, nothing) ===
        fallbackmatrix
    for onesided in (rowsonly, columnsonly)
        @test size(onesided) == size(matrix)
        @test eltype(onesided) == eltype(matrix)

        fill!(rowblock, 0)
        AdaptiveCrossApproximation.nextrc!(rowblock, onesided, rows[1], columns)
        @test rowblock ≈ fullblock[1:1, :]

        fill!(columnblock, 0)
        AdaptiveCrossApproximation.nextrc!(columnblock, onesided, rows, columns[1])
        @test columnblock ≈ fullblock[:, 1:1]

        fill!(block, 0)
        AdaptiveCrossApproximation.nextrc!(block, onesided, rows, columns)
        @test block ≈ fullblock

        fill!(scalarblock, 0)
        AdaptiveCrossApproximation.nextrc!(scalarblock, onesided, rows[1], columns[1])
        @test scalarblock[1] ≈ fullblock[1, 1]
    end

    # Sampled indices need not lie in the prepared index set of the other side.
    otherrows = collect(20:24)
    othercolumns = collect(30:33)
    reference = zeros(ComplexF64, length(rows), length(othercolumns))
    matrix(reference, rows, othercolumns)
    for (j, column) in enumerate(othercolumns)
        fill!(columnblock, 0)
        AdaptiveCrossApproximation.nextrc!(columnblock, rowsonly, rows, column)
        @test columnblock ≈ reference[:, j:j]
    end
    reference = zeros(ComplexF64, length(otherrows), length(columns))
    matrix(reference, otherrows, columns)
    for (i, row) in enumerate(otherrows)
        fill!(rowblock, 0)
        AdaptiveCrossApproximation.nextrc!(rowblock, columnsonly, row, columns)
        @test rowblock ≈ reference[i:i, :]
    end
end

@testset "Far quadstrat and local BEAST KernelMatrix 2D" begin
    line = meshsegment(1.0, 0.05)
    testspace = lagrangec0d1(line)
    trialspace = lagrangec0d1(translate(line, [0.0, 2.0]))
    extension_module = Base.get_extension(AdaptiveCrossApproximation, :ACABEAST)
    rows = collect(2:6)
    columns = collect(8:15)

    for operator in (
        Helmholtz2D.singlelayer(; wavenumber=k),
        Helmholtz2D.doublelayer(; wavenumber=k),
        Helmholtz2D.hypersingular(; wavenumber=k),
    )
        quadstrat = BEAST.defaultquadstrat(operator, testspace, trialspace)
        farquadstrat = AdaptiveCrossApproximation.defaultfarmatrixdata(
            operator, testspace, trialspace
        )
        @test quadstrat isa BEAST.DoubleNumSauterQstrat
        @test farquadstrat ==
            BEAST.DoubleNumQStrat(quadstrat.outer_rule, quadstrat.inner_rule)
        @test AdaptiveCrossApproximation.tofarquadstrat(quadstrat) == farquadstrat

        # Well-separated elements: the far quadrature reproduces the default one.
        reference = assemble(operator, testspace, trialspace; quadstrat=quadstrat)
        matrix = AdaptiveCrossApproximation.AbstractKernelMatrix(
            operator, testspace, trialspace; matrixdata=farquadstrat
        )
        fullblock = zeros(eltype(matrix), length(rows), length(columns))
        matrix(fullblock, rows, columns)
        @test fullblock ≈ reference[rows, columns]

        for (localrows, localcolumns) in
            ((rows, columns), (rows, nothing), (nothing, columns))
            localmatrix = AdaptiveCrossApproximation.localkernelmatrix(
                matrix, localrows, localcolumns
            )
            @test localmatrix isa extension_module.LocalBEASTKernelMatrix

            rowblock = zeros(eltype(matrix), 1, length(columns))
            AdaptiveCrossApproximation.nextrc!(rowblock, localmatrix, rows[2], columns)
            @test rowblock ≈ fullblock[2:2, :]

            columnblock = zeros(eltype(matrix), length(rows), 1)
            AdaptiveCrossApproximation.nextrc!(columnblock, localmatrix, rows, columns[3])
            @test columnblock ≈ fullblock[:, 3:3]
        end
    end
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
