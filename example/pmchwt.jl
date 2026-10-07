using LinearAlgebra
using CompScienceMeshes
using BEAST
using ParallelKMeans
using H2Trees
using AdaptiveCrossApproximation
using PlotlyBase

const ACA = AdaptiveCrossApproximation

# Geometry and function space: dielectric sphere (exterior/interior contrast)
Γ = meshsphere(1.0, 0.08)
X = raviartthomas(Γ)
@show 2 * numfunctions(X)

builder = KMeansTreeBuilder(; numberofclusters=2, minvalues=200)
cluster = KMeansTree(X; builder=builder)
tree = BlockTree(cluster, cluster)

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
    ACA.assemble(op, testspace, trialspace; tree=tree, tol=1e-3, maxrank=60, kwargs...)

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

# Bistatic RCS in the plane φ=0. The far field contains contributions from both
# equivalent currents.
Θ = range(0; stop=π, length=181)
pts = [point(sin(θ), 0, cos(θ)) for θ in Θ]
ffj = potential(MWFarField3D(; wavenumber=κ), pts, u[j], X)
ffm = potential(MWFarField3D(; wavenumber=κ), pts, u[m], X)
ff = -im * κ / (4π) * (-η * ffj + cross.(pts, ffm))
λ = 2π / κ
rcs_dB = 10 .* log10.(4π .* norm.(ff) .^ 2 ./ λ^2)

fcr_j, geo = facecurrents(u[j], X)
fcr_m, _ = facecurrents(u[m], X)

plt = Plot(
    Layout(
        Subplots(;
            rows=1, cols=3, specs=[Spec() Spec(; kind="mesh3d") Spec(; kind="mesh3d")]
        );
        title_text="PMCHWT: dielectric sphere scattering (ACA.assemble)",
    ),
)
add_trace!(
    plt, scatter(; x=rad2deg.(Θ), y=rcs_dB, name="bistatic RCS / λ² [dB]"); row=1, col=1
)
add_trace!(plt, patch(geo, norm.(fcr_j); caxis=(0, 2)); row=1, col=2)
add_trace!(plt, patch(geo, norm.(fcr_m); caxis=(0, 2)); row=1, col=3)

plt
