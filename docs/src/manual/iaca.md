# Using Incomplete Adaptive Cross Approximation

Incomplete Adaptive Cross Approximation (IACA) selects row and column pivots while
evaluating fewer matrix entries than ACA. It is intended primarily for constructing
nested approximations and returns pivot indices rather than materialized low-rank
factors.

## Basic Usage

The convenience constructor uses maximum-value row pivoting, geometric column
pivoting, and extrapolated convergence:

```@example manual-iaca
using AdaptiveCrossApproximation
using LinearAlgebra
using Random
using StaticArrays

Random.seed!(1)
tpos = [SVector(rand(), rand(), rand()) for _ in 1:36]
spos = [SVector(rand(), rand(), rand()) for _ in 1:40] .+
       Ref(SVector(3.5, 0.0, 0.0))

kwave = 3.0
A = ComplexF64[
    (r = norm(t - s); cis(kwave * r) / r) for t in tpos, s in spos
]

tol = 1e-4
maxrank = 25
compressor = IACA(tpos, spos)
rowidcs = collect(axes(A, 1))
colidcs = collect(axes(A, 2))
built = compressor(rowidcs, colidcs, maxrank)

rowbuffer = zeros(eltype(A), maxrank, size(A, 2))
colbuffer = zeros(eltype(A), size(A, 1), maxrank)
rowpivots = zeros(Int, maxrank)
columnpivots = zeros(Int, maxrank)

rank, rows, columns = built(
    A,
    colbuffer,
    rowbuffer,
    rowpivots,
    columnpivots,
    rowidcs,
    colidcs,
    maxrank,
)
(rank, length(rows), length(columns))
```

The returned `rows` and `columns` are global indices into `A`. The work buffers do
not contain a finished pair of low-rank factors as they do for ACA.

## Geometric Row Pivoting

The convenience constructor selects columns geometrically. Reverse the two
pivoting strategies to select rows geometrically instead:

```@example manual-iaca
rowgeometric = IACA(
    MimicryPivoting(spos, tpos),
    MaximumValue(),
    FNormExtrapolator(tol),
)
nothing #hide
```

The first position collection passed to `MimicryPivoting` describes the reference
side and the second contains the candidate pivots. Consequently, their order also
reverses when changing the geometric direction.

## Configuring Pivoting and Convergence

The convenience constructor is equivalent to:

```@example manual-iaca
compressor = IACA(
    MaximumValue(),
    MimicryPivoting(tpos, spos),
    FNormExtrapolator(1e-4),
)
nothing #hide
```

Exactly one direction must use geometric mimicry pivoting while the other uses
[`MaximumValue`](@ref). The geometric strategy chooses a row or column without
evaluating the complementary residual vector; maximum-value pivoting then selects
the pivot in the evaluated vector.

### Pivoting for Nested Blocks

[`MimicryPivoting`](@ref) operates on a complete, identity-indexed candidate range.
For nested blocks whose candidates are represented by tree nodes, use
[`TreeMimicryPivoting`](@ref):

```@example manual-iaca
using H2Trees

builder = TwoNTreeBuilder(; minhalfsize=0.0, minvalues=8)
tree = TwoNTree(spos; builder=builder)
treepivoting = TreeMimicryPivoting(tpos, spos, tree)
treecompressor = IACA(
    MaximumValue(),
    treepivoting,
    FNormExtrapolator(tol),
)
nothing #hide
```

Tree mimicry resolves the supplied candidate nodes to global indices and is the
appropriate strategy inside nested matrix construction. By scoring clusters and
descending through the tree, it avoids repeatedly searching the complete candidate
set and keeps pivot selection practical for large candidate sets. It can additionally
be combined with a [`PivotingFilter`](@ref), such as
[`EFIEDirectionalFilter`](@ref), when the basis functions require directional
selection.

### Convergence Criteria

The standard IACA criterion extrapolates the decay of the sampled residual-vector
norms:

```@example manual-iaca
extrapolated = FNormExtrapolator(1e-4)
directional = PhaseExtrapolator(1e-4)
nothing #hide
```

`FNormExtrapolator` is the general IACA criterion. `PhaseExtrapolator` additionally
tracks direction-specific histories and is intended for directionally filtered
tree-mimicry pivoting. Both contain the norm estimator needed by the incomplete
algorithm; `FNormEstimator` alone is not a complete IACA convergence criterion.

## Skeleton Approximation

IACA returns indices because nested algorithms reuse them directly. For inspection,
they define a CUR-style skeleton approximation:

```@example manual-iaca
approximation = A[:, columns] * (A[rows, columns] \ A[rows, :])
norm(approximation - A) / norm(A)
```

The linear solve is preferable to explicitly forming `inv(A[rows, columns])`. The
skeleton is primarily a diagnostic here; hierarchical and nested algorithms normally
continue working with the selected indices.

For the incomplete algorithm, geometric pivot selection, and its convergence
estimate, see the [IACA theory](../details/iaca.md).
