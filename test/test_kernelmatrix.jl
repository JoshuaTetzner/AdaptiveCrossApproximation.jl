using AdaptiveCrossApproximation
using BEAST
using CompScienceMeshes
using CUDA
using Test

@testset "KernelMatrix" begin
    Γ = meshicosphere(2, 1.0)
    x = lagrangec0d1(Γ)
    y = lagrangec0d1(Γ)
    op = Helmholtz3D.singlelayer()

    A = AdaptiveCrossApproximation.AbstractKernelMatrix(op, x, y)
    P = AdaptiveCrossApproximation.AbstractKernelMatrix(
        (x, y) -> sum(x + y), Γ.vertices, Γ.vertices
    )

    struct kernelfct end
    Base.eltype(::kernelfct) = Float64
    PFct = AdaptiveCrossApproximation.AbstractKernelMatrix(
        kernelfct(), Γ.vertices, Γ.vertices
    )

    @test size(A) == size(P) == size(PFct) == (length(x), length(y))
    @test eltype(A) == eltype(P) == eltype(PFct) == scalartype(op) == Float64

    Γ = meshicosphere(2, Float32(1.0))
    x = lagrangec0d1(Γ)
    y = lagrangec0d1(Γ)
    op = Helmholtz3D.singlelayer(; gamma=Float32(1.0))

    A = AdaptiveCrossApproximation.AbstractKernelMatrix(op, x, y)
    P = AdaptiveCrossApproximation.AbstractKernelMatrix(
        (x, y) -> sum(x + y), Γ.vertices, Γ.vertices
    )
    struct kernelfct32 end
    Base.eltype(::kernelfct32) = Float32
    PFct = AdaptiveCrossApproximation.AbstractKernelMatrix(
        kernelfct32(), Γ.vertices, Γ.vertices
    )

    @test size(A) == size(P) == size(PFct) == (length(x), length(y))
    @test eltype(A) == eltype(P) == eltype(PFct) == scalartype(op) == Float32
end

@testset "GPU BEAST KernelMatrix extension" begin
    extension_module = Base.get_extension(AdaptiveCrossApproximation, :ACABEASTCUDA)
    @test extension_module !== nothing

    Γ = meshicosphere(2, 1.0)
    x = lagrangec0d1(Γ)
    y = lagrangec0d1(Γ)
    op = Helmholtz3D.singlelayer()
    quadstrat = BEAST.DoubleNumQStrat(1, 2)

    cpu_matrix = AdaptiveCrossApproximation.AbstractKernelMatrix(
        op, x, y; matrixdata=quadstrat
    )
    gpu_matrix = AdaptiveCrossApproximation.AbstractKernelMatrix(
        op, x, y; matrixdata=AdaptiveCrossApproximation.GPUMatrixData(quadstrat, 1)
    )

    @test gpu_matrix isa AdaptiveCrossApproximation.GPUBEASTKernelMatrix
    @test size(gpu_matrix) == size(cpu_matrix) == (length(x), length(y))
    @test eltype(gpu_matrix) == eltype(cpu_matrix) == Float64

    rows = collect(1:3)
    columns = collect(1:4)
    cpu_block = zeros(3, 4)
    gpu_block = zeros(3, 4)
    cpu_matrix(cpu_block, rows, columns)
    gpu_matrix(gpu_block, rows, columns)
    @test gpu_block ≈ cpu_block
end

@testset "GPU BEAST KernelMatrix batched block assembly" begin
    # `assemble_blocks_sparse`/`assemble_blocks` at the ACA level were never
    # directly exercised before (only the plain call operator above) - this
    # closes that gap, since assemble_blocks_sparse's generic default now
    # forwards straight to assemble_blocks, which is the one batched GPU
    # entry point (see ext/ACABEASTCUDA/nearinteractions.jl).
    Γ = meshicosphere(3, 1.0)
    x = lagrangec0d1(Γ)
    y = lagrangec0d1(Γ)
    op = Helmholtz3D.singlelayer()
    quadstrat = BEAST.DoubleNumQStrat(1, 2)

    cpu_matrix = AdaptiveCrossApproximation.AbstractKernelMatrix(
        op, x, y; matrixdata=quadstrat
    )
    gpu_matrix = AdaptiveCrossApproximation.AbstractKernelMatrix(
        op, x, y; matrixdata=AdaptiveCrossApproximation.GPUMatrixData(quadstrat)
    )

    n = min(numfunctions(x), 24)
    a = collect(1:n÷2)
    b = collect(n÷2+1:n)
    rowidcs = [a, a, b]
    colidcs = [a, b, b]

    T = eltype(gpu_matrix)
    cpu_blocks = [zeros(T, length(rowidcs[i]), length(colidcs[i])) for i in eachindex(rowidcs)]
    gpu_blocks = [zeros(T, length(rowidcs[i]), length(colidcs[i])) for i in eachindex(rowidcs)]

    for i in eachindex(rowidcs)
        cpu_matrix(cpu_blocks[i], rowidcs[i], colidcs[i])
    end
    AdaptiveCrossApproximation.assemble_blocks_sparse(gpu_matrix, gpu_blocks, rowidcs, colidcs)

    for i in eachindex(rowidcs)
        @test gpu_blocks[i] ≈ cpu_blocks[i]
    end

    # assemble_blocks itself (called directly, as NestedCrossApproximation's
    # far-field batched-bottom-up/coupling-matrix code does) must agree too.
    gpu_blocks2 = [zeros(T, length(rowidcs[i]), length(colidcs[i])) for i in eachindex(rowidcs)]
    AdaptiveCrossApproximation.assemble_blocks(gpu_matrix, gpu_blocks2, rowidcs, colidcs)
    for i in eachindex(rowidcs)
        @test gpu_blocks2[i] ≈ cpu_blocks[i]
    end
end
