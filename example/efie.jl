using LinearAlgebra
using CompScienceMeshes
using BEAST
using ParallelKMeans
using H2Trees
using AdaptiveCrossApproximation
using Krylov
using PlotlyJS

Γ = meshsphere(1.0, 0.08);
X = raviartthomas(Γ);

κ, η = 1.0, 1.0;
t = Maxwell3D.singlelayer(; wavenumber=κ);
E = Maxwell3D.planewave(; direction=ẑ, polarization=x̂, wavenumber=κ);

ttree = KMeansTree(X.pos, 2; minvalues=100)
tree = BlockTree(ttree, ttree)

# Here we permute the space in place, if not familiar with this hmatrix routine be careful
T = HMatrix(t, X, X, tree; tol=1e-3, maxrank=40, isnear=AdaptiveCrossApproximation.isnear())
e = assemble((n × E) × n, X);

u, stats = Krylov.gmres(T, e; rtol=1e-4, verbose=1)
Φ, Θ = [0.0], range(0; stop=π, length=100);
pts = [point(cos(ϕ) * sin(θ), sin(ϕ) * sin(θ), cos(θ)) for ϕ in Φ for θ in Θ];
ffd = potential(MWFarField3D(; wavenumber=κ), pts, u, X);

fcr, geo = facecurrents(u, X);

ys = range(-2; stop=2, length=50);
zs = range(-4; stop=4, length=100);
gridpoints = [point(0, y, z) for y in ys, z in zs];
Esc = potential(MWSingleLayerField3D(; wavenumber=κ), gridpoints, u, X);
Ein = E.(gridpoints);

plt = Plot(
    Layout(
        Subplots(;
            rows=2, cols=2, specs=[Spec() Spec(; rowspan=2); Spec(; kind="mesh3d") missing]
        ),
    ),
)
add_trace!(plt, scatter(; x=Θ, y=norm.(ffd)); row=1, col=1)
add_trace!(
    plt,
    contour(;
        x=zs,
        y=ys,
        z=norm.(Esc - Ein)',
        colorscale="Viridis",
        zmin=0,
        zmax=2,
        showscale=false,
    );
    row=1,
    col=2,
)
add_trace!(plt, patch(geo, norm.(fcr); caxis=(0, 2)); row=2, col=1)

savefig(plt, "efie_results.html"); #hide
nothing #hide
