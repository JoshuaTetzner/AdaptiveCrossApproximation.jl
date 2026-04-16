# General Usage

This page shows how to configure AdaptiveCrossApproximation components in practice.
The examples focus on building blocks that you can combine for ACA, iACA, and
hierarchical matrix assembly.

## ACA Setup

```julia
using AdaptiveCrossApproximation

compressor = ACA(
	rowpivoting=MaximumValue(),
	columnpivoting=MaximumValue(),
	convergence=FNormEstimator(1e-4),
)

U, V = aca(A; maxrank=40, svdrecompress=false)
```

## 2. Convergence Criteria

### Frobenius Norm Estimator

```julia
conv = FNormEstimator(1e-5)
compressor = ACA(convergence=conv)
```

### Incomplete Frobenius Norm Estimator (for iACA workflows)

```julia
iconv = iFNormEstimator(1e-5)
```

### Extrapolation Criterion

```julia
conv_extrap_aca = FNormExtrapolator(1e-5)
conv_extrap_iaca = FNormExtrapolator(iFNormEstimator(1e-5))
```

### Random Sampling Criterion

```julia
conv_rs = RandomSampling(tol=1e-4, factor=1.0)
# Alternative with explicit sample count:
# conv_rs = RandomSampling(tol=1e-4, nsamples=200)
```

### Combined Criterion

```julia
conv_combined = AdaptiveCrossApproximation.CombinedConvCrit([
	FNormEstimator(1e-4),
	RandomSampling(tol=5e-4, factor=1.0),
])
```

## 3. Pivoting Strategies

### Value-Based Pivoting

```julia
rp = MaximumValue()
cp = MaximumValue()

compressor = ACA(rowpivoting=rp, columnpivoting=cp)
```

### Geometry-Based Pivoting

Assume geometric positions are available as vectors of static vectors:

```julia
# tpos and spos are typically Vector{SVector{D,Float64}}
rp_fill = FillDistance(tpos)
rp_leja = Leja2(tpos)
cp_mimic = MimicryPivoting(tpos, spos)
```

### Tree-Aware Geometry Pivoting

```julia
# tree must provide the tree interface expected by TreeMimicryPivoting
cp_tree_mimic = TreeMimicryPivoting(tpos, spos, tree)
```

### Combined Pivoting (advanced)

```julia
piv_combined = AdaptiveCrossApproximation.CombinedPivStrat([
	MaximumValue(),
	AdaptiveCrossApproximation.RandomSamplingPivoting(2),
])
```

## 4. iACA Setup

The package provides a convenience constructor for iACA:

```julia
iaca_default = iACA(tpos, spos)
```

You can also build a custom iACA configuration:

```julia
iaca_custom = iACA(
	MaximumValue(),
	MimicryPivoting(tpos, spos),
	FNormExtrapolator(iFNormEstimator(1e-4)),
)
```

## 5. HMatrix Assembly

High-level entry point:

```julia
hmat = H.assemble(
	operator,
	testspace,
	trialspace;
	tol=1e-4,
	maxrank=40,
	compressor=ACA(tol=1e-4),
	isnear=isnear(1.0),
)
```

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
- iACA for geometric problems: `iACA(MaximumValue(), MimicryPivoting(spos, tpos), FNormExtrapolator(iFNormEstimator(1e-4)))`
- Robust stopping on noisy kernels: combine `FNormEstimator` and `RandomSampling` in `CombinedConvCrit`

For detailed background and theory, see:

- [ACA](../details/aca.md)
- [iACA](../details/iaca.md)
- [Pivoting Strategies](../details/pivoting.md)
- [Convergence Criteria](../details/convergence.md)
- [Hierarchical Matrices](../details/hmatrix.md)
