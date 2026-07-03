using LinearAlgebra
using CompScienceMeshes
using BEAST
using ParallelKMeans
using H2Trees
using AdaptiveCrossApproximation

const ACA = AdaptiveCrossApproximation

# Geometry and function space: dielectric sphere (exterior/interior contrast)
Γ = meshsphere(1.0, 0.1)
X = raviartthomas(Γ)

# Problem setup: PMCHWT for a dielectric sphere illuminated by a plane wave.
# Exterior wavenumber/impedance κ, η; interior κ′, η′.
κ, η = 1.0, 1.0
κ′, η′ = sqrt(5.0) * κ, η / sqrt(5.0)

T = Maxwell3D.singlelayer(; wavenumber=κ)
T′ = Maxwell3D.singlelayer(; wavenumber=κ′)
K = Maxwell3D.doublelayer(; wavenumber=κ)
K′ = Maxwell3D.doublelayer(; wavenumber=κ′)

E = Maxwell3D.planewave(; direction=ẑ, polarization=x̂, wavenumber=κ)
H = -1 / (im * κ * η) * curl(E)

e = (n × E) × n
h = (n × H) × n

# Routes every operator block through ACA's compressed H-matrix assembly (all of
# T, T′, K, K′ are integral operators here); `ACA.assemble` dispatches any local
# operator straight to BEAST's own dense assembly instead, without a manual check.
materialize(op, testspace, trialspace; kwargs...) =
    ACA.assemble(op, testspace, trialspace; tol=1e-3, maxrank=60, kwargs...)

@hilbertspace j m
@hilbertspace k l

α, α′ = 1 / η, 1 / η′
a = (
    η * T[k, j] + η′ * T′[k, j] - K[k, m] - K′[k, m] +
    K[l, j] +
    K′[l, j] +
    α * T[l, m] +
    α′ * T′[l, m]
)
rhs = -e[k] - h[l]

𝕏 = X × X

A = assemble(a, 𝕏, 𝕏; materialize=materialize)
b = assemble(rhs, 𝕏)

A⁻¹ = BEAST.GMRESSolver(A; reltol=1e-4, maxiter=1000)
u = A⁻¹ * b
nothing #hide
