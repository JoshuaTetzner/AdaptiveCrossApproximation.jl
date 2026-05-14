using LinearAlgebra
using CompScienceMeshes
using BEAST
using ParallelKMeans
using H2Trees
using AdaptiveCrossApproximation
using Krylov
using PlotlyJS

Γ = meshsphere(1.0, 0.04);
X = raviartthomas(Γ);
@show numfunctions(X)

κ, η = 2pi, 1.0;
t = Maxwell3D.singlelayer(; wavenumber=κ);
E = Maxwell3D.planewave(; direction=ẑ, polarization=x̂, wavenumber=κ);
Eₜ = (n × E) × n;

ttree = KMeansTree(X.pos, 2; minvalues=100)
tree = BlockTree(ttree, ttree)

function hassemble(op, X, Y; kwargs...)
    return HMatrix(op, X, Y, tree;
        tol=1e-4,
        maxrank=40,
        isnear=AdaptiveCrossApproximation.isnear(),
        spaceordering=AdaptiveCrossApproximation.PreserveSpaceOrder(),
    )
end

@hilbertspace k
@hilbertspace j

a = t[k,j]
A = assemble(a, ∏(X), ∏(X); materialize=hassemble);
b = assemble(Eₜ[k], ∏(X));

A⁻¹ = BEAST.GMRESSolver(A; reltol=1e-4, maxiter=1000)
u = A⁻¹ * b
u = BEAST.FEMFunction(u, ∏(X))

Plot(mesh3d(u[j], colorscale=:Viridis))


