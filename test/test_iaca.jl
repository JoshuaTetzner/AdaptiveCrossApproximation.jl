using AdaptiveCrossApproximation
using H2Trees
using LinearAlgebra
using Random
using StaticArrays
using Test

# IACA returns only pivot indices (rows, cols), never a materialized U/V factorization
# (unlike ACA, whose colbuffer/rowbuffer buffers *are* the finished factors). To check
# accuracy we therefore build the CUR-style skeleton factorization implied by those
# pivots ourselves, mirroring how test_aca.jl checks norm(U*V - A)/norm(A).
curerror(A, rows, cols) = norm(A[:, cols] * inv(A[rows, cols]) * A[rows, :] - A) / norm(A)

@testset "IACA MimicryPivoting (no tree)" begin
    Random.seed!(1234)

    tpos = [@SVector rand(3) for _ in 1:36]
    spos = [@SVector rand(3) for _ in 1:40] .+ Scalar(SVector(3.5, 0.0, 0.0))
    kwave = 3.0
    A = ComplexF64[(r=norm(tp - sp); cis(kwave * r) / r) for tp in tpos, sp in spos]
    maxrank = 25
    rowidcs = collect(1:size(A, 1))
    colidcs = collect(1:size(A, 2))
    tol = 1e-3

    convergences = [
        AdaptiveCrossApproximation.FNormExtrapolator(tol),
        AdaptiveCrossApproximation.FNormExtrapolator(
            AdaptiveCrossApproximation.FNormEstimator(tol)
        ),
    ]

    @testset "geometric columns, MaximumValue rows" begin
        for convergence in convergences
            iaca = IACA(MaximumValue(), MimicryPivoting(tpos, spos), convergence)
            iaca = iaca([1], [1], maxrank)
            rowbuffer = zeros(eltype(A), maxrank, size(A, 2))
            colbuffer = zeros(eltype(A), size(A, 1), maxrank)
            rowpivs = zeros(Int, maxrank)
            colpivs = zeros(Int, maxrank)

            npivot, rows, cols = iaca(
                A, colbuffer, rowbuffer, rowpivs, colpivs, rowidcs, colidcs, maxrank
            )

            @test 0 < npivot <= maxrank
            @test length(unique(rows)) == npivot
            @test length(unique(cols)) == npivot
            @test curerror(A, rows, cols) < 20tol
        end
    end

    @testset "geometric rows, MaximumValue columns" begin
        for convergence in convergences
            iaca = IACA(MimicryPivoting(spos, tpos), MaximumValue(), convergence)
            iaca = iaca([1], [1], maxrank)
            rowbuffer = zeros(eltype(A), maxrank, size(A, 2))
            colbuffer = zeros(eltype(A), size(A, 1), maxrank)
            rowpivs = zeros(Int, maxrank)
            colpivs = zeros(Int, maxrank)

            npivot, rows, cols = iaca(
                A, colbuffer, rowbuffer, rowpivs, colpivs, rowidcs, colidcs, maxrank
            )

            @test 0 < npivot <= maxrank
            @test length(unique(rows)) == npivot
            @test length(unique(cols)) == npivot
            @test curerror(A, rows, cols) < 20tol
        end
    end

    @testset "IACA(tpos, spos) convenience constructor" begin
        iaca = IACA(tpos, spos)
        iaca = iaca([1], [1], maxrank)
        rowbuffer = zeros(eltype(A), maxrank, size(A, 2))
        colbuffer = zeros(eltype(A), size(A, 1), maxrank)
        rowpivs = zeros(Int, maxrank)
        colpivs = zeros(Int, maxrank)

        npivot, rows, cols = iaca(
            A, colbuffer, rowbuffer, rowpivs, colpivs, rowidcs, colidcs, maxrank
        )

        @test 0 < npivot <= maxrank
        @test length(unique(rows)) == npivot
        @test length(unique(cols)) == npivot
        @test curerror(A, rows, cols) < 20 * 1e-4
    end
end

@testset "IACA TreeMimicryPivoting" begin
    Random.seed!(4321)

    tpos = [@SVector rand(3) for _ in 1:60]
    spos = [@SVector rand(3) for _ in 1:64] .+ Scalar(SVector(3.5, 0.0, 0.0))
    kwave = 3.0
    A = ComplexF64[(r=norm(tp - sp); cis(kwave * r) / r) for tp in tpos, sp in spos]
    maxrank = 25
    rowidcs = collect(1:size(A, 1))
    colidcs = collect(1:size(A, 2))
    tol = 1e-3

    treeoncols = TwoNTree(spos, 0.0; minvalues=8)
    treeonrows = TwoNTree(tpos, 0.0; minvalues=8)

    frontier(tree) = collect(H2Trees.LevelIterator(tree, 2))

    @testset "geometric columns (tree), MaximumValue rows" begin
        iaca = IACA(
            MaximumValue(),
            TreeMimicryPivoting(tpos, spos, treeoncols),
            AdaptiveCrossApproximation.FNormExtrapolator(tol),
        )
        iaca = iaca([1], [1], maxrank)
        rowbuffer = zeros(eltype(A), maxrank, size(A, 2))
        colbuffer = zeros(eltype(A), size(A, 1), maxrank)
        rowpivs = zeros(Int, maxrank)
        colpivs = zeros(Int, maxrank)
        colfrontier = frontier(treeoncols)

        npivot, rows, cols = iaca(
            A, colbuffer, rowbuffer, rowpivs, colpivs, rowidcs, colfrontier, maxrank
        )

        @test 0 < npivot <= maxrank
        @test length(unique(rows)) == npivot
        @test length(unique(cols)) == npivot
        @test curerror(A, rows, cols) < 20tol
    end

    @testset "geometric rows (tree), MaximumValue columns" begin
        iaca = IACA(
            TreeMimicryPivoting(spos, tpos, treeonrows),
            MaximumValue(),
            AdaptiveCrossApproximation.FNormExtrapolator(tol),
        )
        iaca = iaca([1], [1], maxrank)
        rowbuffer = zeros(eltype(A), maxrank, size(A, 2))
        colbuffer = zeros(eltype(A), size(A, 1), maxrank)
        rowpivs = zeros(Int, maxrank)
        colpivs = zeros(Int, maxrank)
        rowfrontier = frontier(treeonrows)

        npivot, rows, cols = iaca(
            A, colbuffer, rowbuffer, rowpivs, colpivs, rowfrontier, colidcs, maxrank
        )

        @test 0 < npivot <= maxrank
        @test length(unique(rows)) == npivot
        @test length(unique(cols)) == npivot
        @test curerror(A, rows, cols) < 20tol
    end
end
