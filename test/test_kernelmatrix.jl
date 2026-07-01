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

    @test gpu_matrix isa extension_module.GPUBEASTKernelMatrix
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
