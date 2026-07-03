# General Usage

This page shows how to configure AdaptiveCrossApproximation components in practice.
The examples focus on building blocks that you can combine for ACA, IACA, and
hierarchical matrix assembly. Most code blocks on this page are executed when the
documentation is built, so they stay in sync with the actual API.

## 1. ACA Setup

```@example manual
using AdaptiveCrossApproximation
using LinearAlgebra
using StaticArrays
using Random

Random.seed!(1)
tpos = [SVector(rand(), rand(), rand()) for _ in 1:36]
spos = [SVector(rand(), rand(), rand()) for _ in 1:40] .+ Ref(SVector(3.5, 0.0, 0.0))
A = [inv(norm(t - s)) for t in tpos, s in spos]

compressor = ACA(
    rowpivoting=MaximumValue(),
    columnpivoting=MaximumValue(),
    convergence=FNormEstimator(1e-6),
)

U, V = AdaptiveCrossApproximation.aca(A; tol=1e-6)
norm(U * V - A) / norm(A)
```

Note that the low-level `ACA` struct is exported, but the high-level convenience
function is not — call it as `AdaptiveCrossApproximation.aca(...)`.

### Column-First Variant (ACAᵀ)

`ACAᵀ` is the exact dual of `ACA`: it selects a column first, then a row, alternating
from there. Applying `ACA` to $\bm A$ and `ACAᵀ` to $\bm A^\text{T}$ pick transposed
pivots and yield transposed factors:

```@example manual
Uᵀ, Vᵀ = AdaptiveCrossApproximation.acaᵀ(Matrix(A'); tol=1e-6)
U ≈ Vᵀ', V ≈ Uᵀ'
```

## 2. Convergence Criteria

### Frobenius Norm Estimator

```@example manual
conv = FNormEstimator(1e-5)
compressor2 = ACA(convergence=conv)
nothing #hide
```

### Extrapolation Criterion

`FNormExtrapolator` wraps a `FNormEstimator` and smooths the stopping decision by
fitting the residual-norm decay to a quadratic and extrapolating it. The same type
is used for both ACA (matrix arguments) and IACA (vector arguments, see
[4. IACA Setup](@ref)).

```@example manual
conv_extrap_aca = FNormExtrapolator(1e-5)
conv_extrap_iaca = FNormExtrapolator(FNormEstimator(1e-5))  # equivalent to the line above
nothing #hide
```

### Random Sampling Criterion

```@example manual
conv_rs = AdaptiveCrossApproximation.RandomSampling(; tol=1e-4, factor=1.0)
# Alternative with an explicit sample count:
# conv_rs = AdaptiveCrossApproximation.RandomSampling(; tol=1e-4, nsamples=200)
nothing #hide
```

### Combined Criterion

```@example manual
conv_combined = AdaptiveCrossApproximation.CombinedConvCrit([
	FNormEstimator(1e-4),
	AdaptiveCrossApproximation.RandomSampling(; tol=5e-4, factor=1.0),
])
nothing #hide
```

## 3. Pivoting Strategies

### Value-Based Pivoting

```@example manual
rp = MaximumValue()
cp = MaximumValue()

compressor3 = ACA(rowpivoting=rp, columnpivoting=cp)
nothing #hide
```

### Geometry-Based Pivoting

Using the `tpos`/`spos` positions from [1. ACA Setup](@ref) (row/test positions and
column/trial positions respectively):

```@example manual
rp_fill = FillDistance(tpos)
rp_leja = Leja2(tpos)
cp_mimic = MimicryPivoting(tpos, spos)  # refpos matches the row domain, pos the column domain
nothing #hide
```

### Tree-Aware Geometry Pivoting

```julia
# tree must provide the tree interface expected by TreeMimicryPivoting
# (the ACAH2Trees extension supplies a ready-made adapter for H2Trees.TwoNTree)
cp_tree_mimic = TreeMimicryPivoting(tpos, spos, tree)
```

### Combined Pivoting (advanced)

```@example manual
piv_combined = AdaptiveCrossApproximation.CombinedPivStrat([
	MaximumValue(),
	AdaptiveCrossApproximation.RandomSamplingPivoting(2),
])
nothing #hide
```

## 4. IACA Setup

The package provides a convenience constructor for IACA:

```@example manual
iaca_default = IACA(tpos, spos)
nothing #hide
```

which combines `MaximumValue` row pivoting with `MimicryPivoting` column pivoting
under an `FNormExtrapolator` criterion — equivalent to building it explicitly:

```@example manual
iaca_custom = IACA(
	MaximumValue(),
	MimicryPivoting(tpos, spos),
	FNormExtrapolator(1e-4),
)
nothing #hide
```

Since `MimicryPivoting` has no tree to descend, it is only valid over the *full*
(identity) row/column range — see [`TreeMimicryPivoting`](@ref) for nested/sub-block
compression, as used internally by hierarchical matrix assembly.

Unlike `ACA`, `IACA` returns only the selected pivots, not a finished `U`/`V`
factorization; building it requires a placeholder build call before the real one
(the placeholder indices are discarded and replaced on the first real call):

```@example manual
kwave = 3.0
Awave = ComplexF64[(r = norm(t - s); cis(kwave * r) / r) for t in tpos, s in spos]
maxrank = 25
rowidcs, colidcs = collect(1:size(Awave, 1)), collect(1:size(Awave, 2))

built = iaca_default([1], [1], maxrank)  # placeholder build to obtain typed functors
rowbuffer = zeros(eltype(Awave), maxrank, size(Awave, 2))
colbuffer = zeros(eltype(Awave), size(Awave, 1), maxrank)
rowpivs, colpivs = zeros(Int, maxrank), zeros(Int, maxrank)

npivot, rows, cols = built(
	Awave, colbuffer, rowbuffer, rowpivs, colpivs, rowidcs, colidcs, maxrank
)

# the CUR-style skeleton factorization implied by the returned pivots:
norm(Awave[:, cols] * inv(Awave[rows, cols]) * Awave[rows, :] - Awave) / norm(Awave)
```

## 5. HMatrix Assembly

High-level entry point:

```julia
hmat = AdaptiveCrossApproximation.assemble(
	operator,
	testspace,
	trialspace;
	tol=1e-4,
	maxrank=40,
	compressor=ACA(tol=1e-4),
	isnear=isnear(1.0),
)
```

`AdaptiveCrossApproximation.assemble` picks its tree backend automatically from what's loaded: a
`H2Trees.TwoNTree` when only H2Trees.jl is loaded, or a k-means clustered tree
when ParallelKMeans.jl is loaded alongside it. It also accepts a `quadstrat`
keyword (matching `BEAST.assemble`'s convention), which resolves into
`nearquadstrat=quadstrat` for dense near-field blocks and
`farquadstrat=tofarquadstrat(quadstrat)` for compressed far-field blocks —
override either independently for finer control.

For operators whose matrix is cheap to build directly and never benefits from
low-rank compression (e.g. BEAST's local operators `Identity`/`NCross`), the
`ACABEAST` extension adds a more specific `assemble` method that bypasses tree
construction and ACA entirely, calling `BEAST.assemble` directly.

If you already have a tree, use the explicit constructor:

```julia
hmat = HMatrix(
	operator,
	testspace,
	trialspace,
	tree;
	compressor=ACA(tol=1e-4),
	maxrank=40,
	isnear=isnear(1.0),
)
```

Useful post-processing helpers:

```julia
hf = farmatrix(hmat)      # far-field only
hn = nearmatrix(hmat)     # near-field only
s = storage(hmat)         # storage stats in GB
```

## 6. Recommended Starting Configurations

- Standard ACA: `ACA(rowpivoting=MaximumValue(), columnpivoting=MaximumValue(), convergence=FNormEstimator(1e-4))`
- IACA for geometric problems: `IACA(MaximumValue(), MimicryPivoting(tpos, spos), FNormExtrapolator(1e-4))`
- Robust stopping on noisy kernels: combine `FNormEstimator` and `RandomSampling` in `CombinedConvCrit`

For detailed background and theory, see:

- [ACA](../details/aca.md)
- [IACA](../details/iaca.md)
- [Pivoting Strategies](../details/pivoting.md)
- [Convergence Criteria](../details/convergence.md)
- [Hierarchical Matrices](../details/hmatrix.md)
