# EFIE Scattering from a PEC Sphere

Solves a PEC sphere scattering problem with the Electric Field Integral Equation
(EFIE). A k-means cluster tree is constructed from the RWG basis positions and passed
to `assemble`. A hierarchical LU factorization solves the first-kind EFIE directly;
its accuracy depends on the compression tolerance and conditioning.

The script is `example/efie.jl`. The plots show the bistatic radar cross section normalized by λ² for a
unit-amplitude incident electric field. They are pre-rendered from the script.

```julia
using LinearAlgebra
using CompScienceMeshes
using BEAST
using ParallelKMeans
using H2Trees
using AdaptiveCrossApproximation
using PlotlyBase

const ACA = AdaptiveCrossApproximation
# Run Julia with multiple threads; keep the small block BLAS calls single-threaded.
BLAS.set_num_threads(1)
# Geometry and function space: PEC sphere
Γ = meshsphere(1.0, 0.06)
X = raviartthomas(Γ)
@show numfunctions(X)

builder = KMeansTreeBuilder(; numberofclusters=2, minvalues=100)
cluster = KMeansTree(X; builder=builder)
tree = BlockTree(cluster, cluster)

# Problem setup: EFIE for a PEC sphere illuminated by a plane wave
κ, η = 1.0, 1.0
t = Maxwell3D.singlelayer(; wavenumber=κ)
E = Maxwell3D.planewave(; direction=ẑ, polarization=x̂, wavenumber=κ)

# Assemble the compressed EFIE operator using the shared block tree.
T = ACA.assemble(t, X, X; tree=tree, tol=1e-5, maxrank=100)
e = assemble((n × E) × n, X)

u = T \ e

# Normalized bistatic RCS in the plane φ=0 for unit-amplitude incidence
Θ = range(0; stop=π, length=181)
pts = [point(sin(θ), 0, cos(θ)) for θ in Θ]
ffj = potential(MWFarField3D(; wavenumber=κ), pts, u, X)
ff = im * κ * η / (4π) * ffj
λ = 2π / κ
rcs_dB = 10 .* log10.(4π .* norm.(ff) .^ 2 ./ λ^2)

# Near-field heatmap: total-field magnitude in the y-z plane
ys = range(-2; stop=2, length=60)
zs = range(-3; stop=3, length=120)
gridpoints = [point(0, y, z) for y in ys, z in zs]
Epot = potential(MWSingleLayerField3D(; wavenumber=κ), gridpoints, u, X)
Ein = E.(gridpoints)
Etot = norm.(Epot - Ein)

fcr, geo = facecurrents(u, X)

plt = Plot(
    Layout(
        Subplots(;
            rows=2, cols=2, specs=[Spec() Spec(; rowspan=2); Spec(; kind="mesh3d") missing]
        );
        title_text="EFIE: PEC sphere scattering (ACA.assemble)",
    ),
)
add_trace!(
    plt, scatter(; x=rad2deg.(Θ), y=rcs_dB, name="bistatic RCS / λ² [dB]"); row=1, col=1
)
add_trace!(
    plt,
    contour(; x=zs, y=ys, z=Etot, colorscale="Viridis", showscale=true, name="|E_total|");
    row=1,
    col=2,
)
add_trace!(plt, patch(geo, norm.(fcr); caxis=(0, 2)); row=2, col=1)

plt
```

Normalized bistatic RCS (top left), near-field magnitude in the ``yz`` plane (top right), and
surface-current magnitude (bottom):

```@raw html
<iframe src="../../assets/examples/efie_results.html"
        style="width:100%;height:600px;border:none;"
        loading="lazy">
</iframe>
```

## Notes

- `assemble(t, X, X; tol=..., maxrank=...)` returns an `HMatrix`; `T \ e` factorizes
  it with `lu` and solves. Keep `F = lu(T)` to solve for several right-hand sides.
- The same workflow can be adapted to larger meshes by tuning `tol`, `maxrank`, and
  the H2Trees cluster-splitting thresholds.
