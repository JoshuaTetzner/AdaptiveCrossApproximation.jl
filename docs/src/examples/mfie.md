# MFIE Scattering from a PEC Sphere

Solves the PEC sphere problem of the [EFIE example](efie.md) with the
(better-conditioned, second-kind) Magnetic Field Integral Equation. The system
combines a compressible integral operator (`K`, the Maxwell double layer) with a
local operator (`N`, the surface cross product). The example adapter assembles
`K + 0.5N` into one H-matrix. It caches the integral quadrature data and keeps `N`
sparse while sampling blocks. One shared cluster tree indexes the corresponding RWG
and BC basis functions, which is required by the hierarchical LU.

The script is `example/mfie.jl`. The example-only adapter `beast_hmatrix.jl` in `example/` caches the integral
assemblers and combines the terms before compression, without materializing a dense
matrix. Run the script from `example/` so that the adapter can be included. The plots show the bistatic radar cross section normalized by λ² for a
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
BLAS.set_num_threads(1)
isdefined(@__MODULE__, :BEMHMatrixExample) || include(joinpath(@__DIR__, "beast_hmatrix.jl"))
using .BEMHMatrixExample
# Geometry and function spaces: PEC sphere, primal (RWG) and dual (BC) meshes
Γ = meshsphere(1.0, 0.06)
# Use the same explicit edge ordering for the primal and dual bases.
edges = skeleton(Γ, 1; sort=:spacefillingcurve)
X = raviartthomas(Γ, edges)
Y = buffachristiansen(Γ; edges)
@show numfunctions(X)

builder = KMeansTreeBuilder(; numberofclusters=2, minvalues=100)
# RWG and BC functions are associated with the same primal edges. LU needs
# one shared cluster tree for their coefficient indices. BC positions are
# the primal-edge midpoints; RWG's stored positions use adjacent face centers.
cluster = KMeansTree(BEAST.positions(Y); builder=builder)
tree = BlockTree(cluster, cluster)

# Problem setup: MFIE for a PEC sphere illuminated by a plane wave
ϵ, μ, ω = 1.0, 1.0, 1.0
κ, η = ω * sqrt(ϵ * μ), sqrt(μ / ϵ)

K = Maxwell3D.doublelayer(; wavenumber=κ)
N = BEAST.NCross()
E = Maxwell3D.planewave(; direction=ẑ, polarization=x̂, wavenumber=κ)
H = -1 / (im * μ * ω) * curl(E)
h = (n × H) × n

# Compress K + 0.5N as one HMatrix, not as a sum of matrix-vector maps. The
# adapter caches integral quadrature and assembles the local N only once.
terms = ((1.0, GalerkinTerm(K, Y, X)), (0.5, GalerkinTerm(N, Y, X)))
kernel = GalerkinOperator(((terms,),), numfunctions(X))
A = ACA.HMatrix(
    kernel, Y, X, tree; tol=1e-5, maxrank=100, nearmatrixdata=:near, farmatrixdata=:far
)
b = assemble(h, Y)

u = A \ b

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
        title_text="MFIE: PEC sphere scattering (ACA.assemble)",
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
<iframe src="../../assets/examples/mfie_results.html"
        style="width:100%;height:600px;border:none;"
        loading="lazy">
</iframe>
```

## Notes

- `GalerkinOperator` is an example-only sampling adapter, not a new solver.
  The complete MFIE matrix is factorized by the package's `lu` implementation.
- The original RWG/BC coefficient ordering is retained through assembly and solves.
