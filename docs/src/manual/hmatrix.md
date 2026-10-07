# Using Hierarchical Matrices

An [`HMatrix`](@ref) stores near-field interactions as dense blocks and compresses
admissible far-field interactions with a low-rank compressor. It supports matrix-vector
products, transpose and adjoint products, and conversion to a dense `Matrix`.

## Basic Usage

For BEAST operators, [`AdaptiveCrossApproximation.assemble`](@ref) constructs the
cluster tree and quadrature data automatically:

```julia
using AdaptiveCrossApproximation
using BEAST
using CompScienceMeshes
using H2Trees

mesh = CompScienceMeshes.meshsphere(1.0, 0.3)
space = raviartthomas(mesh)
operator = Maxwell3D.singlelayer(; wavenumber=1.0)

hmat = AdaptiveCrossApproximation.assemble(
    operator,
    space,
    space;
    tol=1e-4,
    maxrank=40,
)
```

The result contains the directly assembled near field and the ACA-compressed far
field. The operator-specific default compressor is used unless `compressor` is
provided explicitly. For supported BEAST double-layer operators, this is the
[robust ACA configuration](aca.md#Robust-ACA-for-Integral-Equations) combining
maximum-value and random-sampling pivoting.

BEAST local operators such as `Identity` and `NCross` are assembled directly and
do not produce an `HMatrix`, because their sparse local interactions do not benefit
from hierarchical low-rank compression.

## Point-Kernel Matrices

Point kernels provide a self-contained way to experiment with hierarchical matrices
without a boundary-element space or quadrature rule. Define a callable kernel with
an `eltype`, cluster its points, and pass both to `HMatrix`:

```julia
using AdaptiveCrossApproximation
using H2Trees
using LinearAlgebra
using ParallelKMeans
using StaticArrays

struct SoftenedKernel{T}
    radius::T
end

Base.eltype(::SoftenedKernel{T}) where {T} = T
(kernel::SoftenedKernel)(x, y) =
    inv(sqrt(sum(abs2, x - y) + kernel.radius^2))

points = [
    SVector(cos(t), sin(t), 0.2sin(2t))
    for t in range(0, 2pi; length=129)[1:(end - 1)]
]
builder = KMeansTreeBuilder(; numberofclusters=2, minvalues=16)
cluster = KMeansTree(points; builder=builder)
tree = BlockTree(cluster, cluster)
kernel = SoftenedKernel(0.05)

hmat = HMatrix(
    kernel,
    points,
    points,
    tree;
    tol=1e-4,
    maxrank=30,
)

dense = [kernel(x, y) for x in points, y in points]
norm(Matrix(hmat) - dense) / norm(dense)
```

The vector-space constructor creates an
[`AdaptiveCrossApproximation.PointMatrix`](@ref), which evaluates only the entries
requested during near-field assembly and ACA compression. This makes point kernels
useful both as toy problems and as a lightweight interface for kernel methods on
point clouds.

## Configuring Assembly

### Compression and Admissibility

The most important compression controls are:

```julia
hmat = AdaptiveCrossApproximation.assemble(
    operator,
    space,
    space;
    tol=1e-4,
    maxrank=40,
    isnear=AdaptiveCrossApproximation.isnear(1.0),
)
```

`tol` controls each far-block compressor and `maxrank` places a hard limit on its
rank. Reaching `maxrank` can indicate that the requested tolerance was not attained.
The admissibility parameter controls which cluster pairs enter the far field; for
the H2Trees backend, increasing it classifies more interactions as far field.

Pass `compressor=ACA(...)` to override the operator-specific default. In particular,
avoid replacing the robust default for a double-layer operator unless the alternative
has been validated for that problem.

### Cluster-Tree Backend

For general use, the k-means backend is the recommended choice. Its balanced,
geometry-adapted clusters usually provide better block partitions, particularly for
large or nonuniformly distributed spaces. For BEAST assembly, load ParallelKMeans
together with H2Trees to select it automatically:

```julia
using ParallelKMeans

hmat = AdaptiveCrossApproximation.assemble(
    operator,
    space,
    space;
    treekwargs=(; numberofclusters=2, minvalues=100),
    tol=1e-4,
    maxrank=40,
)
```

`treekwargs` is forwarded only when the tree is constructed automatically. Larger
leaf sizes reduce tree depth and construction overhead but create larger terminal
blocks. Smaller leaves expose more compression opportunities at the cost of more
blocks and tree traversal. The `TwoNTree` backend remains a useful lightweight
choice for small examples and regularly distributed spaces when ParallelKMeans is
not loaded.

### Supplying an Existing Tree

When a compatible block tree already exists, pass it to the explicit constructor:

```julia
hmat = HMatrix(
    operator,
    testspace,
    trialspace,
    tree;
    tol=1e-4,
    maxrank=40,
    compressor=ACA(; tol=1e-4),
    isnear=AdaptiveCrossApproximation.isnear(1.0),
    spaceordering=AdaptiveCrossApproximation.PreserveSpaceOrder(),
)
```

This entry point is useful when several operators share the same cluster structure
or when the application controls tree construction itself.

### Near- and Far-Field Quadrature

The `quadstrat` keyword follows the convention of `BEAST.assemble`:

```julia
quadrature = BEAST.defaultquadstrat(operator, space, space)
hmat = AdaptiveCrossApproximation.assemble(
    operator,
    space,
    space;
    quadstrat=quadrature,
    tol=1e-4,
)
```

The strategy is used directly for dense near-field blocks. For BEAST's default
Wilton--Sauter strategy, `tofarquadstrat` extracts the nonsingular far-field rules
because singular quadrature is unnecessary for well-separated interactions. Use
`nearquadstrat` and `farquadstrat` when the two parts require explicit, different
strategies.

### Space Ordering and Parallel Assembly

The default `PreserveSpaceOrder()` leaves the test and trial spaces unchanged and
stores their tree indices with the blocks. `PermuteSpaceInPlace()` instead reorders
the spaces to follow the tree layout:

```julia
hmat = AdaptiveCrossApproximation.assemble(
    operator,
    space,
    space;
    spaceordering=AdaptiveCrossApproximation.PermuteSpaceInPlace(),
)
```

The latter mutates the supplied spaces and should only be selected when the caller
expects this reordered layout. Assembly uses a dynamic thread scheduler by default.
Kernel backends may implement `localkernelmatrix` to create independent block-local
assembly caches while sharing the global kernel matrix between threads.

## Working with the Result

`HMatrix` follows Julia's matrix interface:

```julia
x = ones(eltype(hmat), size(hmat, 2))
xt = ones(eltype(hmat), size(hmat, 1))
y = hmat * x
yt = transpose(hmat) * xt
yh = adjoint(hmat) * xt
```

Converting to a dense matrix is useful for validation of small problems but defeats
the storage advantage for large ones:

```julia
A = Matrix(hmat)
```

The near- and far-field contributions can be inspected separately:

```julia
near = AdaptiveCrossApproximation.nearmatrix(hmat)
far = AdaptiveCrossApproximation.farmatrix(hmat)
stored_gigabytes = AdaptiveCrossApproximation.storage(hmat)
```

For block partitioning, admissibility, storage complexity, and the assembly model,
see the [H-matrix theory](../details/hmatrix.md).
