using LinearAlgebra
using CompScienceMeshes
using BEAST
using ParallelKMeans
using H2Trees
using AdaptiveCrossApproximation
using PlotlyBase

const ACA = AdaptiveCrossApproximation
BLAS.set_num_threads(1)
isdefined(@__MODULE__, :BEMHMatrixExample) ||
    include(joinpath(@__DIR__, "beast_hmatrix.jl"))
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
# Colocated components can create zero-radius leaves. Keep coincident cluster
# pairs dense (including the LU diagonal); use the default criterion elsewhere.
lu_admissibility = let geometric = ACA.isnear()
    (rowtree, coltree, row, col) ->
        H2Trees.center(rowtree, row) == H2Trees.center(coltree, col) ||
        geometric(rowtree, coltree, row, col)
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
    kernel,
    𝕏,
    𝕏,
    tree;
    tol=1e-5,
    maxrank=100,
    nearmatrixdata=:near,
    farmatrixdata=:far,
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
