using LinearAlgebra
using CompScienceMeshes
using BEAST
using ParallelKMeans
using H2Trees
using AdaptiveCrossApproximation
using Krylov
using PlotlyJS

Γ = meshsphere(1.0, 0.06);
RT = raviartthomas(Γ);
X = RT × RT

κ, η = 2pi, 1.0;
T = Maxwell3D.singlelayer(; wavenumber=κ);
K = Maxwell3D.doublelayer(; wavenumber=κ);
N = BEAST.NCross();

κ′, η′ = 2.0 * κ,  η
T′ = Maxwell3D.singlelayer(; wavenumber=κ′);
K′ = Maxwell3D.doublelayer(; wavenumber=κ′);

E = Maxwell3D.planewave(; direction=ẑ, polarization=x̂, wavenumber=κ);
H = -1/(im*μ) * curl(E)

e = (n × E) × n;
h = (n × H) × n;


function materialize(op, X, Y; kwargs...)
    ttree = KMeansTree(X.pos, 2; minvalues=100)
    tree = BlockTree(ttree, ttree)
    H = HMatrix(op, X, Y, tree;
        tol=1e-4,
        maxrank=40,
        isnear=AdaptiveCrossApproximation.isnear(),
        spaceordering=AdaptiveCrossApproximation.PreserveSpaceOrder(),
    )
    AdaptiveCrossApproximation.storage(H)
    return H
end

@hilbertspace p q
@hilbertspace m j

α, α′ = 1/η, 1/η′
a = (
    α*T[p,m]+α′*T′[p,m] + K[p,j]+K′[p,j]
    -K[q,m]-K′[q,m] + η*T[q,j]+η′*T′[q,j]
)
b = -h[p] + e[q] 

𝐀 = assemble(a, X, X; materialize=materialize);
𝐛 = assemble(b, X);

𝐀⁻¹ = BEAST.GMRESSolver(𝐀; reltol=1e-4, maxiter=1000)
𝐮 = 𝐀⁻¹ * 𝐛
𝐮 = BEAST.FEMFunction(𝐮, X)

Plot(mesh3d(𝐮[j], colorscale=:Viridis))


