using BEAST
using CompScienceMeshes
using AdaptiveCrossApproximation
using Test

# Float64

Γ = meshicosphere(2, 1.0)
x = lagrangec0d1(Γ)
y = lagrangec0d1(Γ)
op = Helmholtz3D.singlelayer()

A = AdaptiveCrossApproximation.AbstractKernelMatrix(op, x, y)
P = AdaptiveCrossApproximation.AbstractKernelMatrix(
    (x, y) -> sum(x + y), Γ.vertices, Γ.vertices
)

struct myfct end
Base.eltype(::myfct) = Float64
PFct = AdaptiveCrossApproximation.AbstractKernelMatrix(myfct(), Γ.vertices, Γ.vertices)

@test size(A) == size(P) == size(PFct) == (length(x), length(y))
@test eltype(A) == eltype(P) == eltype(PFct) == scalartype(op) == Float64

# Float32

Γ = meshicosphere(2, Float32(1.0))
x = lagrangec0d1(Γ)
y = lagrangec0d1(Γ)
op = Helmholtz3D.singlelayer(; gamma=Float32(1.0))

A = AdaptiveCrossApproximation.AbstractKernelMatrix(op, x, y)
P = AdaptiveCrossApproximation.AbstractKernelMatrix(
    (x, y) -> sum(x + y), Γ.vertices, Γ.vertices
)
struct myfct32 end
Base.eltype(::myfct32) = Float32
PFct = AdaptiveCrossApproximation.AbstractKernelMatrix(myfct32(), Γ.vertices, Γ.vertices)

@test size(A) == size(P) == size(PFct) == (length(x), length(y))
@test eltype(A) == eltype(P) == eltype(PFct) == scalartype(op) == Float32
