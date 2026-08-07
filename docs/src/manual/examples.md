# Application Examples

This section contains end-to-end examples that show how AdaptiveCrossApproximation
is used in realistic BEAST scattering workflows. Each example assembles a boundary
integral operator as a compressed H-matrix through `AdaptiveCrossApproximation.assemble`,
solves the resulting linear system, and produces two plots: a near-field heatmap and a
bistatic radar cross section (RCS) pattern, alongside the induced surface current.

These snippets are based on the files in `example/` (`efie.jl`, `mfie.jl`, `pmchwt.jl`).
The EFIE and MFIE plots below are pre-rendered from those scripts (via
`docs/render_examples.jl`) rather than executed on every docs build, since a full
BEAST + PlotlyJS scattering solve is too heavy for that; run the scripts directly to
reproduce or modify them.

## EFIE Scattering from a PEC Sphere

Solves a PEC sphere scattering problem with the Electric Field Integral Equation
(EFIE). `assemble` builds the H2Trees cluster tree automatically from the RWG
basis positions, so no manual tree construction is needed.

```julia
using LinearAlgebra
using CompScienceMeshes
using BEAST
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

ACA.storage(T)

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
        ),
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

savefig(plt, "efie_results.html")
```

Bistatic RCS (top left), near-field magnitude in the ``yz`` plane (top right), and
surface-current magnitude (bottom):

```@raw html
<iframe src="../../assets/examples/efie_results.html"
        style="width:100%;height:600px;border:none;"
        loading="lazy">
</iframe>
```

### Notes

- `assemble(t, X, X; tol=..., maxrank=...)` returns an `HMatrix` compatible with
  `Krylov.gmres` directly (no BEAST `materialize` callback needed for a single operator).
- The same workflow can be adapted to larger meshes by tuning `tol`, `maxrank`, and
  the H2Trees cluster-splitting thresholds.

## MFIE Scattering from a PEC Sphere

Solves the same PEC sphere problem with the (better-conditioned, second-kind)
Magnetic Field Integral Equation. The system combines a compressible integral
operator (`K`, the Maxwell double layer) with a local operator (`N`, the surface
cross product). BEAST needs a `materialize` callback to route each block to the
right assembly routine; `ACA.assemble` dispatches on the operator type itself
(tree + ACA compression for `K`, direct `BEAST.assemble` for the local `N`), so
the callback is a thin pass-through rather than a manual branch.

```julia
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

@hilbertspace j
@hilbertspace m

a = K[m, j] + 0.5 * N[m, j]
l = h[m]

A = assemble(a, ∏(Y), ∏(X); materialize=ACA.assemble)
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
        ),
        title_text="MFIE: PEC sphere scattering (ACA.assemble)",
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

savefig(plt, "mfie_results.html")
```

Bistatic RCS (top left), near-field magnitude in the ``yz`` plane (top right), and
surface-current magnitude (bottom):

```@raw html
<iframe src="../../assets/examples/mfie_results.html"
        style="width:100%;height:600px;border:none;"
        loading="lazy">
</iframe>
```

### Notes

- `assemble(a, ∏(Y), ∏(X); materialize=materialize)` calls `materialize` once per
  bilinear-form block (`K` and `N` here); `ACA.assemble` itself decides whether
  ACA compression applies, based on the operator's type.
- MFIE converges in far fewer GMRES iterations than EFIE since it is a well-conditioned
  second-kind formulation.

## PMCHWT Scattering from a Dielectric Sphere

Solves a homogeneous dielectric sphere scattering problem with the PMCHWT
formulation, which couples exterior (`κ, η`) and interior (`κ′, η′`) traces through
four integral operators (`T`, `T′`, `K`, `K′`), all of which are compressible.

```julia
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

# Routes every operator block through ACA's compressed H-matrix assembly (all of
# T, T′, K, K′ are integral operators here); `ACA.assemble` dispatches any local
# operator straight to BEAST's own dense assembly instead, without a manual check.
materialize(op, testspace, trialspace; kwargs...) =
    ACA.assemble(op, testspace, trialspace; tol=1e-3, maxrank=60, kwargs...)

@hilbertspace j m
@hilbertspace k l

α, α′ = 1 / η, 1 / η′
a = (
    η * T[k, j] + η′ * T′[k, j] - K[k, m] - K′[k, m] + K[l, j] + K′[l, j] +
    α * T[l, m] +
    α′ * T′[l, m]
)
rhs = -e[k] - h[l]

𝕏 = X × X

A = assemble(a, 𝕏, 𝕏; materialize=materialize)
b = assemble(rhs, 𝕏)

A⁻¹ = BEAST.GMRESSolver(A; reltol=1e-4, maxiter=1000)
u = A⁻¹ * b

# Bistatic RCS in the plane φ=0: σ(θ) = 4π|E_far(θ)|² for unit-amplitude incidence.
# The exterior scattered far field combines the electric (j) and magnetic (m)
# equivalent currents the same way the exterior block of `a` does (η*T[j] - K[m]).
Θ = range(0; stop=π, length=181)
pts = [point(sin(θ), 0, cos(θ)) for θ in Θ]
ffd_e = potential(MWFarField3D(; wavenumber=κ), pts, u[j], X)
ffd_m = potential(BEAST.MWDoubleLayerFarField3D(; wavenumber=κ), pts, u[m], X)
rcs_dB = 10 .* log10.(4π .* abs2.(norm.(η .* ffd_e .- ffd_m)))

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
add_trace!(plt, scatter(; x=rad2deg.(Θ), y=rcs_dB, name="bistatic RCS [dB]"); row=1, col=1)
add_trace!(plt, patch(geo, norm.(fcr_j); caxis=(0, 2)); row=1, col=2)
add_trace!(plt, patch(geo, norm.(fcr_m); caxis=(0, 2)); row=1, col=3)

savefig(plt, "pmchwt_results.html")
```

Bistatic RCS (left), electric surface current magnitude `|j|` (middle), and magnetic
surface current magnitude `|m|` (right):

```@raw html
<iframe src="../../assets/examples/pmchwt_results.html"
        style="width:100%;height:450px;border:none;"
        loading="lazy">
</iframe>
```

### Notes

- PMCHWT has two coupled unknowns (electric current `j`, magnetic current `m`); the
  solved hilbert-space vector `u` is indexed per-unknown as `u[j]`, `u[m]`.
