using LinearAlgebra
using CompScienceMeshes
using BEAST
using H2Trees
using AdaptiveCrossApproximation
using PlotlyJS

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

# All PMCHWT operator blocks (T, T′, K, K′) are integral operators, so this
# routes the entire system through ACA's compressed H-matrix assembly.
function materialize(op, testspace, trialspace; kwargs...)
    if op isa BEAST.IntegralOperator
        return ACA.H.assemble(op, testspace, trialspace; tol=1e-3, maxrank=60)
    end
    return BEAST.assemble(op, testspace, trialspace; kwargs...)
end

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

# Bistatic RCS in the plane φ=0. The far field has two contributions (electric
# current j and magnetic current m), combined as in the standard PMCHWT far-field
# formula: σ(θ) = 4π|E_far(θ)|² for unit-amplitude incidence.
Θ = range(0; stop=π, length=181)
pts = [point(sin(θ), 0, cos(θ)) for θ in Θ]
ffm = potential(MWFarField3D(; wavenumber=κ), pts, u[m], X)
ffj = potential(MWFarField3D(; wavenumber=κ), pts, u[j], X)
ff = -η * im * κ * ffj + im * κ * cross.(pts, ffm)
rcs_dB = 10 .* log10.(4π .* abs2.(norm.(ff)))

# Near-field heatmap: total-field magnitude in the y-z plane. Uses the
# equivalence-principle superposition trick: the exterior representation is
# correct (and vanishes) outside (inside) Γ, the interior one vice versa, so
# simply adding both gives the physically correct field everywhere.
function nearfield_E(um, uj, Xm, Xj, κ, η, points, Einc=(x -> point(0, 0, 0)))
    Kfield = MWDoubleLayerField3D(; wavenumber=κ)
    Tfield = MWSingleLayerField3D(; wavenumber=κ)
    Em = potential(Kfield, points, um, Xm)
    Ej = potential(Tfield, points, uj, Xj)
    return -Em + η * Ej + Einc.(points)
end

ys = range(-2; stop=2, length=60)
zs = range(-3; stop=3, length=120)
gridpoints = [point(0, y, z) for y in ys, z in zs]
E_ex = nearfield_E(u[m], u[j], X, X, κ, η, gridpoints, E)
E_in = nearfield_E(-u[m], -u[j], X, X, κ′, η′, gridpoints)
Etot = norm.(E_ex + E_in)

fcr, geo = facecurrents(u[j], X)

plt = Plot(
    Layout(
        Subplots(;
            rows=2, cols=2, specs=[Spec() Spec(; rowspan=2); Spec(; kind="mesh3d") missing]
        );
        title_text="PMCHWT: dielectric sphere scattering (ACA.H.assemble)",
    ),
)
add_trace!(plt, scatter(; x=rad2deg.(Θ), y=rcs_dB, name="bistatic RCS [dB]"); row=1, col=1)
add_trace!(
    plt,
    contour(; x=zs, y=ys, z=Etot, colorscale="Viridis", showscale=true, name="|E_total|");
    row=1,
    col=2,
)
add_trace!(plt, patch(geo, norm.(fcr); caxis=(0, 2)); row=2, col=1)

savefig(plt, "pmchwt_results.html"); #hide
nothing #hide
