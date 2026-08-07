using BEAST
using CompScienceMeshes
using StaticArrays
using PlotlyJS
using AdaptiveCrossApproximation
using LinearAlgebra

λ = 5.0
k = 2 * pi / λ
gamma = im * k
alpha = -gamma
beta = -1 / gamma
op = Maxwell3D.singlelayer(; wavenumber=2 * pi / 5.0)
opA = Maxwell3D.singlelayer(; gamma=gamma, alpha=alpha, beta=0.0 * im)
opϕ = Maxwell3D.singlelayer(; gamma=gamma, alpha=im * 0.0, beta=beta)
## Common plane
##A
m1 = rotate(meshrectangle(1.0, 1.0, 1.0), SVector(0.0, 0.0, π / 4))
m2 = translate(m1, SVector(2.0, 0.0, 0.0))
plot([wireframe(m1), wireframe(m2)])

t = raviartthomas(m1)
s = raviartthomas(m2)
val = assemble(op, t, s)
valA = assemble(opA, t, s)
valϕ = assemble(opϕ, t, s)

##B
m1 = rotate(meshrectangle(1.0, 1.0, 1.0), SVector(0.0, 0.0, π / 4))
m2 = translate(m1, SVector(0.0, 2.0, 0.0))
plot([wireframe(m1), wireframe(m2)])

t = raviartthomas(m1)
s = raviartthomas(m2)
val = assemble(op, t, s)
valA = assemble(opA, t, s)
valϕ = assemble(opϕ, t, s)

##C
m1 = rotate(meshrectangle(1.0, 1.0, 1.0), SVector(0.0, 0.0, π / 4))
m2 = translate(rotate(m1, SVector(0.0, 0.0, pi / 2)), SVector(3.0, sqrt(2.0) / 2, 0.0))
plot([wireframe(m1), wireframe(m2)])

t = raviartthomas(m1)
s = raviartthomas(m2)
val = assemble(op, t, s)
valA = assemble(opA, t, s)
valϕ = assemble(opϕ, t, s)

## Facing planes
#D
m1 = meshrectangle(1.0, 1.0, 1.0)
m2 = translate(m1, SVector(0.0, 0.0, 1.0))
plot([wireframe(m1), wireframe(m2)])
t = raviartthomas(m1)
s = raviartthomas(m2)
val = assemble(op, t, s)
valA = assemble(opA, t, s)
valϕ = assemble(opϕ, t, s)

#E
m1 = meshrectangle(1.0, 1.0, 1.0)
m2 = translate(rotate(m1, SVector(0.0, 0.0, pi / 2)), SVector(1.0, 1.0, 1.0))
plot([wireframe(m1), wireframe(m2)])
t = raviartthomas(m1)
s = raviartthomas(m2)
val = assemble(op, t, s)
valA = assemble(opA, t, s)
valϕ = assemble(opϕ, t, s)

##
λ = 0.5
k = 2 * pi / λ
gamma = im * k
alpha = -gamma
beta = -1 / gamma
op = Maxwell3D.singlelayer(; wavenumber=2 * pi / 5.0)
opA = Maxwell3D.singlelayer(; gamma=gamma, alpha=alpha, beta=0.0 * im)
opϕ = Maxwell3D.singlelayer(; gamma=gamma, alpha=im * 0.0, beta=beta)

##
m1 = meshrectangle(1.0, 1.0, 0.05)
m2 = CompScienceMeshes.translate(m1, SVector(3.0, 0.0, 0.0))
#=m2 = translate(
    weld(
        meshrectangle(1.0, 1.0, 0.1),
        translate(
            rotate(meshrectangle(1.0, 1.0, 0.1), SVector(0.0, π / 2, 0.0)),
            SVector(1.0, 0.0, 0.0),
        ),
    ),
    SVector(3.0, 0.0, 0.0),
)=#
#plot([wireframe(m1), wireframe(m2)])
##
t = raviartthomas(m1)
s = raviartthomas(m2)
blk = assemble(opA, t, s)

U, V = AdaptiveCrossApproximation.aca(blk; tol=1e-3)
norm(blk - U * V) / norm(blk)
