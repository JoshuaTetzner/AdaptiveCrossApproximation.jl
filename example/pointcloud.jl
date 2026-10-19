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
