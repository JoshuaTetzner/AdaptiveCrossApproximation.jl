# Application Examples

This section contains end-to-end examples that show how AdaptiveCrossApproximation
is used in BEAST scattering workflows and point-cloud kernel systems. The scattering
examples assemble the complete operator as a compressed H-matrix and solve it with
`\`, which uses a hierarchical LU factorization. They also construct far-field,
near-field, and surface-current plots.

The examples report the bistatic radar cross section normalized by λ² for a
unit-amplitude incident electric field.

These snippets match the files in `example/` (`efie.jl`, `mfie.jl`, `pmchwt.jl`).
MFIE and PMCHWT use the example-only `beast_hmatrix.jl` adapter in the same directory;
it caches integral assemblers and combines the terms before compression, without
materializing a full dense matrix. Run the snippets from `example/` so this helper
can be included.
The plots below are pre-rendered from those scripts via `docs/render_examples.jl`
rather than recomputed during every documentation build. The plotting commands remain
in each example; saving the documentation assets is kept separate from the examples.

## EFIE Scattering from a PEC Sphere

Solves a PEC sphere scattering problem with the Electric Field Integral Equation
(EFIE). A k-means cluster tree is constructed from the RWG basis positions and passed
to `assemble`. A hierarchical LU factorization solves the first-kind EFIE directly;
its accuracy depends on the compression tolerance and conditioning.

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
ACA.storage(T)

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

### Notes

- `assemble(t, X, X; tol=..., maxrank=...)` returns an `HMatrix`; `T \ e` factorizes
  it with `lu` and solves. Keep `F = lu(T)` to solve for several right-hand sides.
- The same workflow can be adapted to larger meshes by tuning `tol`, `maxrank`, and
  the H2Trees cluster-splitting thresholds.

## MFIE Scattering from a PEC Sphere

Solves the same PEC sphere problem with the (better-conditioned, second-kind)
Magnetic Field Integral Equation. The system combines a compressible integral
operator (`K`, the Maxwell double layer) with a local operator (`N`, the surface
cross product). The example adapter assembles `K + 0.5N` into one H-matrix. It
caches the integral quadrature data and keeps `N` sparse while sampling blocks.
One shared cluster tree indexes the corresponding RWG and BC basis functions,
which is required by the hierarchical LU.

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

### Notes

- `GalerkinOperator` is an example-only sampling adapter, not a new solver.
  The complete MFIE matrix is factorized by the package's `lu` implementation.
- The original RWG/BC coefficient ordering is retained through assembly and solves.

## PMCHWT Scattering from a Dielectric Sphere

Solves a homogeneous dielectric sphere scattering problem with the PMCHWT
formulation, which couples exterior (`κ, η`) and interior (`κ′, η′`) traces through
four integral operators (`T`, `T′`, `K`, `K′`), all of which are compressible.

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

# Geometry and function space: dielectric sphere (exterior/interior contrast)
Γ = meshsphere(1.0, 0.08)
X = raviartthomas(Γ)
@show 2 * numfunctions(X)

builder = KMeansTreeBuilder(; numberofclusters=2, minvalues=100)
# Cluster both currents geometrically in one tree. This allows dense-leaf
# pivoting to couple the colocated electric and magnetic unknowns.
dofpositions = vcat(BEAST.positions(X), BEAST.positions(X))
cluster = KMeansTree(dofpositions; builder=builder)
tree = BlockTree(cluster, cluster)
# Colocated components can create zero-radius leaves. Always keep diagonal
# terminals dense for LU; use the default geometric criterion everywhere else.
lu_admissibility = let geometric = ACA.isnear()
    (rowtree, coltree, row, col) -> row == col || geometric(rowtree, coltree, row, col)
end

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

α, α′ = 1 / η, 1 / η′
# Cache each integral assembler once, including its near/far quadrature data.
t, t′ = GalerkinTerm(T, X, X), GalerkinTerm(T′, X, X)
k, k′ = GalerkinTerm(K, X, X), GalerkinTerm(K′, X, X)
blocks = (
    (((η, t), (η′, t′)), ((-1.0, k), (-1.0, k′))),
    (((1.0, k), (1.0, k′)), ((α, t), (α′, t′))),
)
kernel = GalerkinOperator(blocks, numfunctions(X))
𝕏 = X × X
A = ACA.HMatrix(
    kernel, 𝕏, 𝕏, tree;
    tol=1e-5, maxrank=100, nearmatrixdata=:near, farmatrixdata=:far,
    isnear=lu_admissibility,
)
b = -vcat(assemble(e, X), assemble(h, X))

u = A \ b
# The solution is in the original ordering: electric current first, magnetic second.
uj, um = view(u, 1:numfunctions(X)), view(u, (numfunctions(X) + 1):length(u))

# Bistatic RCS in the plane φ=0. The far field contains contributions from both
# equivalent currents.
Θ = range(0; stop=π, length=181)
pts = [point(sin(θ), 0, cos(θ)) for θ in Θ]
ffj = potential(MWFarField3D(; wavenumber=κ), pts, uj, X)
ffm = potential(MWFarField3D(; wavenumber=κ), pts, um, X)
ff = -im * κ / (4π) * (-η * ffj + cross.(pts, ffm))
λ = 2π / κ
rcs_dB = 10 .* log10.(4π .* norm.(ff) .^ 2 ./ λ^2)

fcr_j, geo = facecurrents(uj, X)
fcr_m, _ = facecurrents(um, X)

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
```

Normalized bistatic RCS (left), electric surface current magnitude `|j|` (middle), and magnetic
surface current magnitude `|m|` (right):

```@raw html
<iframe src="../../assets/examples/pmchwt_results.html"
        style="width:100%;height:450px;border:none;"
        loading="lazy">
</iframe>
```

### Notes

- PMCHWT has two coupled unknowns: electric and magnetic current. The solution is
  stored as `[uj; um]` in the original basis order.
- Both components are clustered together geometrically. Dense-leaf pivoting can
  therefore couple colocated electric and magnetic unknowns.
- The factorization uses pivoting inside dense diagonal leaves, not unrestricted
  global pivoting. Singular diagonal leaves can still prevent a hierarchical LU.


## Coulomb Kernel on a Point Cloud

`example/pointcloud.jl` needs no BEAST or plotting dependencies. It builds a binary
k-means tree with `minvalues=100`, uses the default admissibility, and directly
solves an H-matrix for exact `1/r` interactions with an explicitly zero diagonal.
The fast radius update guarantees monotone parent/child radii, not containment of
child bounding balls. The zero diagonal makes this matrix indefinite; it is a
self-interaction convention, not a regularization or a guarantee of invertibility.

```julia
# Run with multiple Julia threads, e.g. julia --threads=auto example/pointcloud.jl.
using AdaptiveCrossApproximation
using H2Trees
using LinearAlgebra
using ParallelKMeans
using Random
using StaticArrays

const ACA = AdaptiveCrossApproximation
BLAS.set_num_threads(1)
Random.seed!(1234)

struct CoulombKernel end
Base.eltype(::CoulombKernel) = Float64
function (::CoulombKernel)(x, y)
    r = norm(x - y)
    # Exact 1/r off the diagonal; zero is an explicit self-interaction convention.
    return iszero(r) ? 0.0 : inv(r)
end

# Uniformly distributed points on the unit sphere.
npoints = 10_000
points = [normalize(SVector(randn(), randn(), randn())) for _ in 1:npoints]
kernel = CoulombKernel()
builder = KMeansTreeBuilder(;
    numberofclusters=2, minvalues=100, updateradii=H2Trees.unsafemaxradiusboundingsphere
)
cluster = KMeansTree(points; builder)
tree = BlockTree(cluster, cluster)

H = ACA.HMatrix(kernel, points, points, tree; tol=1e-8, maxrank=100)
ACA.storage(H)

b = randn(npoints)
x = H \ b
```
