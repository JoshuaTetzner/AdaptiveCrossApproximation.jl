using LinearAlgebra
using CompScienceMeshes
using BEAST
using ParallelKMeans
using H2Trees
using HFactory: HFactory
using AdaptiveCrossApproximation

# Run Julia with multiple threads; keep the small block BLAS calls single-threaded.
BLAS.set_num_threads(1)

# Geometry and function space: PEC sphere
Γ = meshsphere(1.0, 0.06)
X = raviartthomas(Γ)
N = numfunctions(X)
@show N

# One k-means cluster tree shared by both approaches.
builder = KMeansTreeBuilder(; numberofclusters=2, minvalues=100)
cluster = KMeansTree(X; builder=builder)

# EFIE operator and plane-wave excitation
κ = 1.0
t = Maxwell3D.singlelayer(; wavenumber=κ)
E = Maxwell3D.planewave(; direction=ẑ, polarization=x̂, wavenumber=κ)
e = assemble((n × E) × n, X)

tol, maxrank = 1e-5, 100

# 1) ACA's HMatrix on a block tree of the k-means clusters.
tree = BlockTree(cluster, cluster)
@time "ACA HMatrix" T = AdaptiveCrossApproximation.assemble(
    t, X, X; tree=tree, tol=tol, maxrank=maxrank, verbose=false
)

# 2) HFactory: block index over the same clusters, ACA compression of far blocks.
clusterindex = HFactory.ClusterIndex(cluster, N)
index = HFactory.BlockIndex(clusterindex, clusterindex)
kernel = AbstractKernelMatrix(t, X, X)
scheme = HFactory.HMatrixConstruction(nothing, LowRankCompression(ACA(; tol); maxrank))
@time "HFactory HMatrix" H = HFactory.assemble(kernel, index, scheme)
ws = HFactory.MatVecWorkspace(H)

# Compare storage and products of both compressed operators.
storage(near, far) = sum(length, near) + sum(B -> length(B.U) + length(B.V), far)
compression(near, far) = round(storage(near, far) / N^2; digits=3)
println(
    "ACA:      $(length(T.near)) dense, $(length(T.far)) low-rank blocks, ",
    "storage $(compression(T.near, T.far)) N²",
)
println(
    "HFactory: $(length(H.dense)) dense, $(length(H.compressed)) low-rank blocks, ",
    "storage $(compression(H.dense, H.compressed)) N²",
)

x = randn(ComplexF64, N)
y = similar(x)
mul!(y, T, x), mul!(y, ws, x) # compile before timing
@time "ACA MVP" mul!(y, T, x)
@time "HFactory MVP" mul!(y, ws, x)
@show norm(ws * x - T * x) / norm(T * x)
@show norm(transpose(ws) * x - transpose(T) * x) / norm(transpose(T) * x)

# Solve with ACA's H-matrix LU and check the residual with the HFactory operator.
u = T \ e
@show norm(ws * u - e) / norm(e)
