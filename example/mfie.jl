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
Y = buffachristiansen(Γ);
@show numfunctions(X)

κ, η = 2pi, 1.0;
ϵ, μ = 1/η, η;
t = Maxwell3D.singlelayer(; wavenumber=κ);
E = Maxwell3D.planewave(; direction=ẑ, polarization=x̂, wavenumber=κ);
H = -1/(im*μ) * curl(E)

Eₜ = (n × E) × n;
Hₜ = (n × H) × n;

@hilbertspace k
@hilbertspace j

function materialize(op, X, Y; kwargs...)
    if op isa BEAST.IntegralOperator
        Xtree = KMeansTree(X.pos, 2; minvalues=100)
        Ytree = KMeansTree(Y.pos, 2; minvalues=100)
        tree = BlockTree(Xtree, Ytree)
        return HMatrix(op, X, Y, tree;
            tol=1e-4,
            maxrank=40,
            isnear=AdaptiveCrossApproximation.isnear(),
            spaceordering=AdaptiveCrossApproximation.PreserveSpaceOrder(),
        )
    end
    return BEAST.assemble(op, X, Y; kwargs...)
end

K = Maxwell3D.doublelayer(; wavenumber=κ);
N = BEAST.NCross();

a = K[k,j] + 0.5*N[k,j]
l = Hₜ[k]

A = assemble(a, ∏(Y), ∏(X); materialize=materialize);
b = assemble(l, ∏(Y));

A⁻¹ = BEAST.GMRESSolver(A; reltol=1e-4, maxiter=1000)
u = A⁻¹ * b
u = BEAST.FEMFunction(u, ∏(X))

Plot(mesh3d(u[j], colorscale=:Viridis))

Xtree = KMeansTree(X.pos, 2; minvalues=100)
Ytree = KMeansTree(Y.pos, 2; minvalues=100)
tree = BlockTree(Xtree, Ytree)
@which HMatrix(K, Y, X, tree;
    tol=1e-4,
    maxrank=40,
    isnear=AdaptiveCrossApproximation.isnear(),
    spaceordering=AdaptiveCrossApproximation.PreserveSpaceOrder(),)