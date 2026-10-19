# Using Hierarchical Matrices

An [`HMatrix`](@ref) stores near-field interactions as dense blocks and compresses
admissible far-field interactions with a low-rank compressor. It supports matrix-vector
products, transpose and adjoint products, sums and products of H-matrices, LU
factorization and solves, and conversion to a dense `Matrix`.

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

### [Symmetric Assembly on One Space](@id symmetric-hmatrix-assembly)

For a transpose-symmetric operator, use the single-space call:

```julia
hmat = AdaptiveCrossApproximation.assemble(operator, space; tol=1e-4)
```

With an existing block tree, use the corresponding single-space constructor:

```julia
hmat = HMatrix(operator, space, tree; tol=1e-4)
```

Automatic assembly constructs one shared cluster tree. The explicit constructor
uses only the supplied tree's test cluster tree for both axes. Compression,
admissibility, and quadrature options follow general assembly.

These calls assume that the operator and any custom `isnear` predicate are
transpose-symmetric; symmetry is not checked. For complex matrices the requirement
is `A == transpose(A)`, rather than Hermitian symmetry. The two-space calls
`assemble(operator, space, space)` and `HMatrix(operator, space, space, tree)`
continue to use general assembly, even when both arguments are the same object.

Each unordered terminal interaction is assembled or compressed once. The
opposite orientation references the same physical block and applies a transpose,
without conjugation. Diagonal near blocks are assembled and stored in full, then
symmetrized by copying their upper triangles into their lower triangles. Both
space-ordering modes are supported. Such matrices support products with vectors;
hierarchical arithmetic requires general (full) assembly.

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
the spaces to follow the tree layout in general two-space assembly:

```julia
hmat = AdaptiveCrossApproximation.assemble(
    operator,
    space,
    space;
    spaceordering=AdaptiveCrossApproximation.PermuteSpaceInPlace(),
)
```

The latter mutates the supplied spaces and should only be selected when the caller
expects this reordered layout. Assembly runs on `ntasks` parallel tasks (keyword,
default `Threads.nthreads()`).
Kernel backends may implement `localkernelmatrix` to create independent block-local
assembly caches while sharing the global kernel matrix between threads.

## Working with the Result

`HMatrix` behaves like a matrix in products. For repeated products, construct a
`MatVecWorkspace` once; it holds the matrix, a parallel schedule and all internal
buffers, and is used in place of the matrix:

```julia
A = AdaptiveCrossApproximation.MatVecWorkspace(hmat)
y = A * x
mul!(y, A, x)                   # no numerical allocations once warm
mul!(y, transpose(A), xt, α, β)
Y = A' * X                      # several right-hand sides at once
```

The product is distributed over `ntasks` parallel tasks (keyword, default
`Threads.nthreads()`; `ntasks=1` runs on the calling task). One
workspace serves one product at a time; give each concurrently working task its
own copy, `MatVecWorkspace(A)`, which shares the schedule and the matrix. A real
matrix also multiplies complex vectors. Products with `hmat` itself
(`hmat * x`, `mul!(y, hmat, x)`) build a temporary workspace on every call and
warn once.

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

The assembled object has concrete type [`HMatrixCore`](@ref); `HMatrix` remains
its public compatibility name. The numerical payload is available as
`hmat.near::Vector{Matrix{T}}` and
`hmat.far::Vector{LowRankMatrix{T}}`. `hmat.csr` is the canonical row-cluster
ordering of terminal blocks. `hmat.csc` stores only column-cluster traversal
metadata and integer references back into the CSR entries; it does not duplicate
the matrices. The retained `hmat.tree` supplies the cluster hierarchy and index
sets, and `hmat.tol` the assembly tolerance.

## Hierarchical Arithmetic

Sums, products, recompression and the LU factorization of `HMatrix` objects use
Julia's standard functions. Every result block is truncated to the tolerance
`tol` chosen at assembly; for operands with different tolerances the larger one
is used. Truncation is in the spectral norm, relative to each block. The
[theory section](../details/hmatrix.md#Hierarchical-Arithmetic) explains the
algorithms.

Arithmetic requires full block storage: matrices from [symmetric
assembly](@ref symmetric-hmatrix-assembly) support products with vectors only.

### A Complete Example

The following point-kernel example factorizes a matrix and solves a system with
it. The regularized single-layer kernel gives a well-conditioned system; in
BEM applications, `hmat` comes from `assemble`.

```julia
using AdaptiveCrossApproximation
using H2Trees
using LinearAlgebra
using ParallelKMeans
using StaticArrays

BLAS.set_num_threads(1)

struct RegularizedKernel{T}
    delta::T
end

Base.eltype(::RegularizedKernel{T}) where {T} = T
(kernel::RegularizedKernel)(x, y) = inv(4pi * (norm(x - y) + kernel.delta))

points = [
    SVector(sinpi(θ) * cospi(φ), sinpi(θ) * sinpi(φ), cospi(θ)) for
    (θ, φ) in zip(range(0, 1; length=2000), range(0, 80; length=2000))
]
cluster = KMeansTree(points; builder=KMeansTreeBuilder(; numberofclusters=2, minvalues=32))
tree = BlockTree(cluster, cluster)
kernel = RegularizedKernel(0.01)

hmat = HMatrix(kernel, points, points, tree; tol=1e-8, maxrank=100)

F = lu(hmat)                                         # hmat is unchanged
S = AdaptiveCrossApproximation.SolveWorkspace(F)
b = rand(length(points))
x = S \ b

dense = [kernel(x, y) for x in points, y in points]
norm(dense * x - b) / norm(b)                        # ≈ 6e-10
```

### Solving Systems

```julia
F = lu(hmat)            # factorize a copy; hmat is unchanged
lu!(hmat)               # factorize in place: hmat holds the factors afterwards

S = AdaptiveCrossApproximation.SolveWorkspace(F)
x = S \ b               # vector or matrix of right-hand sides
ldiv!(S, b)             # in place
ldiv!(x, S, b)
x = transpose(S) \ b
x = S' \ b
```

The factorization follows `P*A ≈ L*U`, where `P` permutes rows only inside each
dense diagonal leaf. `lu!` overwrites the blocks of the matrix with the factors
and warns once. An exactly zero pivot in a leaf raises a
[`LeafLUFailure`](@ref); with `check=false`, `issuccess(F)` is `false` instead.

`hmat \ b` factorizes a copy of `hmat` and solves in one step. To solve several
times with the same matrix, keep `F = lu(hmat)` and solve with `F \ b`.

A [`SolveWorkspace`](@ref AdaptiveCrossApproximation.SolveWorkspace) holds the
buffers of the solves and acts like the factorization. Solving with `F`
directly (`F \ b`, `ldiv!(F, b)`) builds a temporary workspace on every call.
The factorization is read-only during solves: tasks solving
concurrently with one factorization each use their own workspace,
`SolveWorkspace(S)`.

### Preconditioning

With a coarse tolerance, the LU factorization is cheap and usually an effective
preconditioner for an iterative solver. For example, with
[Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl):

```julia
using Krylov

coarse = HMatrix(kernel, points, points, tree; tol=1e-2, maxrank=100)
P = AdaptiveCrossApproximation.SolveWorkspace(lu(coarse))
A = AdaptiveCrossApproximation.MatVecWorkspace(hmat)

x, stats = Krylov.gmres(A, b; M=P, ldiv=true, rtol=1e-8)
```

For the example above, this reduces the GMRES iterations from 58 to 4. The
preconditioner needs the same row and column count as `hmat`, but not the same
partition.

### Reusing Workspaces

The convenience calls build the workspace they need. Every operation also
accepts an explicitly constructed workspace as its last argument, which avoids
planning and allocating buffers again:

| Operation | Workspace |
|:--|:--|
| `M * x`, `mul!(y, M, x)` | `MatVecWorkspace(H)`, used in place of `H` |
| `lu!(H, ws)`, `lu(H, ws)` | `LUWorkspace(H)` |
| `S \ b`, `ldiv!(S, b)` | `SolveWorkspace(F)`, used in place of `F` |
| `mul!(C, A, B, α, β, ws)` | `ProductWorkspace(C, A, B)` |
| `axpy!(α, A, C, ws)`, `axpby!(α, A, β, C, ws)` | `AdditionWorkspace(C, A)` |
| `recompress!(H, ws)` | `RecompressionWorkspace(H)` |

All workspace types live in the `AdaptiveCrossApproximation` namespace. A
workspace analyses the partition of the matrices it was built for. Except for
the matrix-vector and solve workspaces, which hold their matrix, it can be
reused for every matrix with the same cluster trees and block partition. Ranks
may differ, for example for the matrices of a frequency sweep on one geometry:

```julia
ws = AdaptiveCrossApproximation.LUWorkspace(H₁)
F₁ = lu!(H₁, ws)
F₂ = lu!(H₂, ws)        # same partition: no new planning
```

The `LUWorkspace` holds the elimination schedule, every triangular-solve and
Schur-update plan and their buffers. Factorizations do not depend on the
workspace after `lu!` returns. One workspace serves one operation at a time.

### Sums and Products

```julia
C = A + B
C = A - 2B
axpy!(α, A, C)          # C = α*A + C
axpby!(α, A, β, C)      # C = α*A + β*C
rmul!(C, α)             # exact scaling

C = A * B
mul!(C, A, B, α, β)     # C = α*A*B + β*C
```

Sums require the same cluster trees and block partition. Products require
matching cluster trees (the column tree of `A` is the row tree of `B`), while
the block partitions may differ; `A * B` has the partition of `A`. In-place
operations overwrite only their destination, which must not be a factor of the
product. Scaling is exact and does not change the ranks.

### Recompression

```julia
recompress!(hmat)
```

ACA does not produce optimal ranks. `recompress!` truncates every low-rank block
to the smallest rank meeting `hmat.tol`, without changing the partition. In the
example above, it reduces the storage by about 10%. It overwrites `hmat` and
warns once; [`AdaptiveCrossApproximation.nnz`](@ref) counts the stored scalars
before and after.

### Accuracy and Diagnostics

The tolerance bounds the error of every single truncation, not the error of a
product or factorization: the errors of many truncations accumulate. Check the
residual of a solve, or the error against a dense reference for small problems,
before relying on a tolerance. A factorization reports its truncations:

```julia
F.converged          # every truncation met the tolerance
F.rank_limited       # some truncation was capped by a rank limit
F.max_block_error    # largest discarded spectral error of a block
F.flushes            # number of truncations
```

### [Threading](@id hmatrix-threading)

Every parallel operation takes the number of tasks as the keyword `ntasks`,
with default `Threads.nthreads()`; `ntasks=1` runs on the calling task. The
scheduling itself is chosen internally for each operation:

```julia
hmat = HMatrix(operator, testspace, trialspace, tree; ntasks=8)
A = AdaptiveCrossApproximation.MatVecWorkspace(hmat; ntasks=8)
AdaptiveCrossApproximation.RecompressionWorkspace(hmat; ntasks=8)
AdaptiveCrossApproximation.LUWorkspace(hmat; ntasks=8)
AdaptiveCrossApproximation.ProductWorkspace(C, A, B; ntasks=8)
AdaptiveCrossApproximation.AdditionWorkspace(C, A; ntasks=8)
```

Assembly, matrix-vector products, sums and recompression consist of
independent pieces of uneven cost; each task takes the next piece as soon as it
is done. The block operations of the LU factorization and of products depend on
each other. Their workspace records once which blocks every operation reads and
writes; the tasks then take the earliest operation whose inputs are final. The
factors and products are identical for every `ntasks`. The forms without a
workspace (`hmat * x`, `lu(hmat)`, `A + B`, `A * B`, `recompress!(hmat)`) use
`Threads.nthreads()` tasks.

Start Julia with several threads and use a single BLAS thread
(`BLAS.set_num_threads(1)`): the block operations are small, and threaded BLAS
calls slow them down.

`example/pointcloud.jl` constructs and directly solves a Coulomb-kernel H-matrix
on a point cloud, including a small dense-reference check.

For block partitioning, admissibility, storage complexity, and the assembly model,
see the [H-matrix theory](../details/hmatrix.md).
