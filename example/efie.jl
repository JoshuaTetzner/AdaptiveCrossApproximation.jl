using LinearAlgebra
using CompScienceMeshes
using BEAST
using ParallelKMeans
using H2Trees
using AdaptiveCrossApproximation
using Krylov
using PlotlyJS

const ACA = AdaptiveCrossApproximation
# Geometry and function space: PEC sphere
Γ = meshsphere(1.0, 0.1)
X = raviartthomas(Γ)

# Problem setup: EFIE for a PEC sphere illuminated by a plane wave
κ, η = 1.0, 1.0
t = Maxwell3D.singlelayer(; wavenumber=κ)
E = Maxwell3D.planewave(; direction=ẑ, polarization=x̂, wavenumber=κ)

# Assemble the compressed EFIE operator directly through ACA's high-level entry
# point. `assemble` builds an H2Trees cluster tree from the RWG basis positions
# and compresses admissible (far) blocks with ACA -- no manual tree needed.
T = ACA.assemble(t, X, X; tol=1e-3, maxrank=60)
e = assemble((n × E) × n, X)

u, stats = Krylov.gmres(T, e; rtol=1e-4)
@assert stats.solved "GMRES failed to converge"
ACA.storage(T); #hide

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
        title_text="EFIE: PEC sphere scattering (ACA.assemble)",
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

savefig(plt, "efie_results.html"); #hide
nothing #hide
