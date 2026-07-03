using LinearAlgebra
using CompScienceMeshes
using BEAST
using H2Trees
using AdaptiveCrossApproximation
using PlotlyJS

const ACA = AdaptiveCrossApproximation

# Geometry and function spaces: PEC sphere, primal (RWG) and dual (BC) meshes
Γ = meshsphere(1.0, 0.1)
X = raviartthomas(Γ)
Y = buffachristiansen(Γ)

# Problem setup: MFIE for a PEC sphere illuminated by a plane wave
ϵ, μ, ω = 1.0, 1.0, 1.0
κ, η = ω * sqrt(ϵ * μ), sqrt(μ / ϵ)

K = Maxwell3D.doublelayer(; wavenumber=κ)
N = BEAST.NCross()
E = Maxwell3D.planewave(; direction=ẑ, polarization=x̂, wavenumber=κ)
H = -1 / (im * μ * ω) * curl(E)
h = (n × H) × n

# Route integral-operator blocks through ACA's compressed H-matrix assembly and
# leave the local (identity-like) NCross block to BEAST's own dense assembly.
function materialize(op, testspace, trialspace; kwargs...)
    if op isa BEAST.IntegralOperator
        return ACA.H.assemble(op, testspace, trialspace; tol=1e-3, maxrank=60)
    end
    return BEAST.assemble(op, testspace, trialspace; kwargs...)
end

@hilbertspace j
@hilbertspace m

a = K[m, j] + 0.5 * N[m, j]
l = h[m]

A = assemble(a, ∏(Y), ∏(X); materialize=materialize)
b = assemble(l, ∏(Y))

A⁻¹ = BEAST.GMRESSolver(A; reltol=1e-4, maxiter=1000)
u = A⁻¹ * b

# Bistatic RCS in the plane φ=0: σ(θ) = 4π|E_far(θ)|² for unit-amplitude incidence
Θ = range(0; stop=π, length=181)
pts = [point(sin(θ), 0, cos(θ)) for θ in Θ]
ffd = potential(MWFarField3D(; wavenumber=κ), pts, u, X)
rcs_dB = 10 .* log10.(4π .* abs2.(norm.(ffd)))

# Near-field heatmap: total-field magnitude in the y-z plane
ys = range(-2; stop=2, length=60)
zs = range(-3; stop=3, length=120)
gridpoints = [point(0, y, z) for y in ys, z in zs]
Esc = potential(MWSingleLayerField3D(; wavenumber=κ), gridpoints, u, X)
Ein = E.(gridpoints)
Etot = norm.(Esc + Ein)

fcr, geo = facecurrents(u, X)

plt = Plot(
    Layout(
        Subplots(;
            rows=2, cols=2, specs=[Spec() Spec(; rowspan=2); Spec(; kind="mesh3d") missing]
        );
        title_text="MFIE: PEC sphere scattering (ACA.H.assemble)",
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

savefig(plt, "mfie_results.html"); #hide
nothing #hide
