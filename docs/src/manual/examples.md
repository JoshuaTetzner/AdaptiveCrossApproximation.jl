# Application Examples

This section contains end-to-end examples that show how
AdaptiveCrossApproximation is used in realistic workflows.

## EFIE Scattering with HMatrix Compression

The example below is based on the repository file `example/efie.jl`.
It solves a PEC sphere scattering problem with the Electric Field Integral Equation (EFIE),
assembles the operator as an H-matrix, and solves the linear system with GMRES.

```julia
using LinearAlgebra
using CompScienceMeshes
using BEAST
using ParallelKMeans
using H2Trees
using AdaptiveCrossApproximation
using Krylov
using PlotlyJS

# Geometry and function space
Γ = meshsphere(1.0, 0.08)
X = raviartthomas(Γ)

# Problem setup
κ, η = 1.0, 1.0
t = Maxwell3D.singlelayer(; wavenumber=κ)
E = Maxwell3D.planewave(; direction=ẑ, polarization=x̂, wavenumber=κ)

# Block tree for H-matrix assembly
ttree = KMeansTree(X.pos, 2; minvalues=100)
tree = BlockTree(ttree, ttree)

# Assemble compressed EFIE matrix
T = HMatrix(
	t,
	X,
	X,
	tree;
	tol=1e-3,
	maxrank=40,
	isnear=AdaptiveCrossApproximation.isnear(),
)

# Right-hand side and linear solve
e = assemble((n × E) × n, X)
u, stats = Krylov.gmres(T, e; rtol=1e-4, verbose=1)

# Far-field pattern
Φ, Θ = [0.0], range(0; stop=π, length=100)
pts = [point(cos(ϕ) * sin(θ), sin(ϕ) * sin(θ), cos(θ)) for ϕ in Φ for θ in Θ]
ffd = potential(MWFarField3D(; wavenumber=κ), pts, u, X)

# Surface currents
fcr, geo = facecurrents(u, X)

# Near-field slice
ys = range(-2; stop=2, length=50)
zs = range(-4; stop=4, length=100)
gridpoints = [point(0, y, z) for y in ys, z in zs]
Esc = potential(MWSingleLayerField3D(; wavenumber=κ), gridpoints, u, X)
Ein = E.(gridpoints)

# Visualisation
plt = Plot(
	Layout(
		Subplots(; rows=2, cols=2, specs=[Spec() Spec(; rowspan=2); Spec(; kind="mesh3d") missing]),
	),
)
add_trace!(plt, scatter(; x=Θ, y=norm.(ffd)); row=1, col=1)
add_trace!(
	plt,
	contour(; x=zs, y=ys, z=norm.(Esc - Ein)', colorscale="Viridis", zmin=0, zmax=2, showscale=false);
	row=1,
	col=2,
)
add_trace!(plt, patch(geo, norm.(fcr); caxis=(0, 2)); row=2, col=1)

savefig(plt, "efie_results.html") #hide
nothing #hide
```

```@raw html
<object data="../efie_results.html" type="text/html"  style="width:100%; height:100vh;"> </object>
```

### Notes

- This uses `HMatrix(...)` directly with a precomputed bounding-ball tree.
- `Krylov.gmres` is used for the iterative solve on the compressed system.
- The same workflow can be adapted to larger meshes by tuning `tol`, `maxrank`, and tree settings.
