# Using Adaptive Cross Approximation

Adaptive Cross Approximation constructs a low-rank factorization from selected rows
and columns of a matrix. Use [`ACA`](@ref) for the row-first algorithm and
[`ACAᵀ`](@ref) for its column-first counterpart.

## Basic Usage

The high-level convenience function returns the two factors directly:

```@example manual-aca
using AdaptiveCrossApproximation
using LinearAlgebra
using StaticArrays

x = range(0.0, 1.0; length=40)
y = range(2.0, 3.0; length=44)
A = [inv(abs(xi - yj)) for xi in x, yj in y]

U, V = aca(A; tol=1e-6)
norm(U * V - A) / norm(A)
```

For repeated compression or integration into another algorithm, construct the
compressor explicitly and provide reusable buffers:

```@example manual-aca
maxrank = 20
compressor = ACA(; tol=1e-6)
colbuffer = zeros(size(A, 1), maxrank)
rowbuffer = zeros(maxrank, size(A, 2))

rank = compressor(A, colbuffer, rowbuffer, maxrank)
U = colbuffer[:, 1:rank]
V = rowbuffer[1:rank, :]
norm(U * V - A) / norm(A)
```

Submatrices can be compressed without copying them by passing their global indices:

```@example manual-aca
rows = collect(1:2:size(A, 1))
cols = collect(1:2:size(A, 2))
fill!(colbuffer, 0)
fill!(rowbuffer, 0)

rank = compressor(
    A,
    view(colbuffer, 1:length(rows), :),
    view(rowbuffer, :, 1:length(cols)),
    maxrank;
    rowidcs=rows,
    colidcs=cols,
)
norm(
    colbuffer[1:length(rows), 1:rank] * rowbuffer[1:rank, 1:length(cols)] -
    A[rows, cols],
) / norm(A[rows, cols])
```

## Column-First ACA

`ACAᵀ` selects a column first and is the exact dual of `ACA`:

```@example manual-aca
Uᵀ, Vᵀ = AdaptiveCrossApproximation.acaᵀ(Matrix(A'); tol=1e-6)
U ≈ Vᵀ', V ≈ Uᵀ'
```

## Configuring Pivoting and Convergence

The default configuration uses maximum-value pivoting in both directions and a
Frobenius-norm estimate:

```@example manual-aca
compressor = ACA(
    rowpivoting=MaximumValue(),
    columnpivoting=MaximumValue(),
    convergence=FNormEstimator(1e-5),
)
nothing #hide
```

### Robust ACA for Integral Equations

For boundary element matrices involving double-layer operators, or whenever the
standard ACA convergence estimate may be unreliable, combine maximum-value
pivoting with pivots obtained from randomly sampled residual entries. The
corresponding convergence criterion combines the Frobenius-norm estimate with the
random-sampling estimate:

```@example manual-aca
tol = 1e-4
normcriterion = FNormEstimator(tol)
samplingcriterion = AdaptiveCrossApproximation.RandomSampling(; tol=tol)
convergence = AdaptiveCrossApproximation.CombinedConvCrit([
    normcriterion,
    samplingcriterion,
])
rowpivoting = AdaptiveCrossApproximation.CombinedPivStrat([
    MaximumValue(),
    AdaptiveCrossApproximation.RandomSamplingPivoting(1),
])

robust = ACA(; rowpivoting=rowpivoting, convergence=convergence)
nothing #hide
```

The algorithm first uses ordinary maximum-value row pivots. If the Frobenius-norm
criterion signals convergence while sampled residual entries remain too large, it
continues with row pivots selected from those samples. Compression stops only when
both criteria are satisfied. This is the recommended robust configuration for the
affected integral equations [[4, 6]](@ref refs). The BEAST extension selects it
automatically for supported double-layer operators.

### Other Pivoting Strategies

The remaining pivoting strategies target more specialized applications:

```@example manual-aca
positions = [SVector(xi) for xi in x]
referencepositions = [SVector(yi) for yi in y]

maximumvalue = MaximumValue()
filldistance = FillDistance(positions)
leja = Leja2(positions)
mimicry = MimicryPivoting(referencepositions, positions)
nothing #hide
```

[`FillDistance`](@ref) and [`Leja2`](@ref) use geometric positions instead of
residual values. [`MimicryPivoting`](@ref) and [`TreeMimicryPivoting`](@ref) are
intended for incomplete and nested constructions; a tree-aware strategy is built
as `TreeMimicryPivoting(referencepositions, positions, tree)`. Their specific use
is described in the [IACA manual](iaca.md).

### Other Convergence Criteria

The convergence criteria can be constructed independently of a compressor:

```@example manual-aca
estimated = FNormEstimator(1e-5)
extrapolated = FNormExtrapolator(1e-5)
phaseextrapolated = PhaseExtrapolator(1e-5)
sampled = AdaptiveCrossApproximation.RandomSampling(; tol=1e-5, factor=1.0)
nothing #hide
```

`FNormEstimator` is the standard ACA criterion. `FNormExtrapolator` also considers
the decay of recent updates. `RandomSampling` monitors selected residual entries
and is most useful in the combined robust configuration above. `PhaseExtrapolator`
is intended for directionally filtered tree-mimicry pivoting rather than ordinary
ACA.

## Full-Pivoted ACA

When the complete dense matrix is already available, [`FullPivoting`](@ref) can
select every pivot from the entire remaining residual matrix:

```@example manual-aca
fullpivoted = ACA(
    FullPivoting(),
    FullPivoting(),
    FNormEstimator(1e-6),
)

workingmatrix = copy(A)
rank, rowpivots, columnpivots = fullpivoted(workingmatrix, min(size(A)...))
approximation = A[:, columnpivots] * (A[rowpivots, columnpivots] \ A[rowpivots, :])
norm(approximation - A) / norm(A)
```

This method overwrites its matrix argument while computing the residual, which is
why the example passes `copy(A)`. It returns the selected row and column indices,
not low-rank factors, and does not use the row/column sampling interface.

For the algorithms and the available combinations of pivoting and convergence
strategies, see the [ACA theory](../details/aca.md).
