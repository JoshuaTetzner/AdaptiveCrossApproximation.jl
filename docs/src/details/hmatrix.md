# Hierarchical Matrices (H-Matrices)

## Introduction

A $\mathcal{H}$-Matrix is a block-structured matrix representation that exploits the low-rank nature of blockwise interactions in many physical problems. Instead of storing the full matrix, an $\mathcal{H}$-Matrix separates the matrix into two complementary components:

- **Near-field blocks**: Stored explicitly as dense submatrices for geometrically close cluster pairs
- **Far-field blocks**: Stored as low-rank factors for geometrically separated cluster pairs

This dual representation significantly reduces memory requirements while maintaining accuracy, especially for large-scale systems arising from integral equations, boundary element methods, and kernel matrix approximations.

## Motivation

For many kernels and operators (e.g., Green's functions in acoustics and electromagnetism), interactions between distant degrees of freedom decay rapidly or become approximately low-rank. An $\mathcal{H}$-Matrix exploits this by:

1. **Hierarchical partitioning**: Recursively dividing the domain into clusters via a tree
2. **Admissibility testing**: Distinguishing well-separated pairs (low-rank) from close pairs (dense)
3. **Selective compression**: Compressing only far-field blocks via low-rank factorization

This approach reduces:
- **Storage**: From $\mathcal{O}(n^2)$ to $\mathcal{O}(kn \log n)$ for rank-$k$ blocks
- **Matrix-vector products**: From $\mathcal{O}(n^2)$ to $\mathcal{O}(kn \log n)$
- **Assembly time**: From $\mathcal{O}(n^2)$ to $\mathcal{O}(kn \log n)$

## Admissibility and Separation

The admissibility condition determines whether a block pair is near or far. Given two clusters with diameters $d_t$ and $d_s$ and geometric distance $\text{dist}$, the standard admissibility criterion is:

$$\text{dist} \geq \eta \cdot \max(d_t, d_s)$$

The dimensionless parameter $\eta$ controls the geometric separation threshold:

- **Small $\eta$ (e.g., $\eta = 1$)**: More aggressive low-rank approximation, stricter separation
- **Large $\eta$**: More dense blocks, potentially higher accuracy but more storage

API: [`AdaptiveCrossApproximation.isnear`](@ref)

## Core Data Structures

### HMatrix

The hierarchical matrix structure stores near and far interactions separately.

```
HMatrix{K, NearType, FarType}
├── nearinteractions: BlockSparseMatrix (explicit dense blocks)
├── farinteractions: Vector[BlockSparseMatrix, ...] (compressed blocks by level)
└── dim: Tuple{Int, Int} (matrix dimensions)
```

API: [`AdaptiveCrossApproximation.HMatrix`](@ref)

### LowRankMatrix

Far-field blocks are stored efficiently as left and right factors to avoid forming the dense product:

$$\bm C = \bm U \bm V^T \approx (m \times r)(r \times n)$$

The low-rank matrix evaluates matrix-vector products as $\bm U(\bm V^T \bm x)$ to minimize operations.

API: [`AdaptiveCrossApproximation.LowRankMatrix`](@ref)

## Assembly Process

The HMatrix assembly pipeline:

```
Operator + Spaces + Tree
    ↓
Admissibility Testing (isnear)
    ├─→ Near-field clusters → Dense evaluation → BlockSparseMatrix
    └─→ Far-field clusters → ACA compression → LowRankMatrix blocks
    ↓
HMatrix(nearinteractions, farinteractions, dimensions)
```

### Near-Field Assembly

For admissible cluster pairs (below separation threshold), the matrix block is computed explicitly and stored as a dense submatrix within a block-sparse structure.

API: [`AdaptiveCrossApproximation.nearinteractions`](@ref), [`AdaptiveCrossApproximation.assemblenears`](@ref)

### Far-Field Assembly

For inadmissible cluster pairs (well-separated), the matrix block is compressed using adaptive cross approximation (ACA) or similar algorithms. The compression typically:

1. Evaluates only a small subset of rows and columns
2. Constructs low-rank factors via ACA
3. Stores factors as `LowRankMatrix` with rank $r \ll \min(m, n)$

API: [`AdaptiveCrossApproximation.farinteractions`](@ref), [`AdaptiveCrossApproximation.assemblefars`](@ref)

### Space Ordering

The assembly process can optionally reorder the test and trial spaces to align with tree partitioning, improving block structure and cache locality.

- **`PermuteSpaceInPlace()`** (default): Reorders spaces in-place via tree permutation
- **`PreserveSpaceOrder()`**: Maintains original space ordering

## High-Level Interface

The main entry point for assembling an HMatrix:

```julia
HMatrix(operator, testspace, trialspace, tree;
    tol=1e-4,
    compressor=ACA(; tol=tol),
    isnear=isnear(1.0),
    maxrank=40,
    spaceordering=PermuteSpaceInPlace(),
    nearmatrixdata=nothing,
    farmatrixdata=nothing,
    scheduler=DynamicScheduler()
)
```

### Key Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `operator` | — | Bilinear form or kernel for matrix entry evaluation |
| `testspace` | — | Row basis or evaluation points |
| `trialspace` | — | Column basis or evaluation points |
| `tree` | — | Hierarchical tree controlling block partitioning |
| `tol` | `1e-4` | Compression tolerance for ACA |
| `compressor` | `ACA(tol=1e-4)` | Compression algorithm (ACA, ACAᵀ, iACA) |
| `isnear` | `isnear(1.0)` | Admissibility predicate with `η = 1.0` |
| `maxrank` | `40` | Hard limit on compressed block rank |
| `spaceordering` | `PermuteSpaceInPlace()` | Strategy for space reordering |

## Access and Analysis

### Extracting Components

Extract near or far contributions separately:

```julia
hmat = HMatrix(operator, testspace, trialspace, tree)
dense_blocks = nearmatrix(hmat)      # BlockSparseMatrix
lowrank_blocks = farmatrix(hmat)     # HMatrix with only far-field
```

API: [`AdaptiveCrossApproximation.nearmatrix`](@ref), [`AdaptiveCrossApproximation.farmatrix`](@ref)

### Storage Analysis

Analyze and report memory requirements:

```julia
storage(hmat)
```

Outputs:
- Total storage in GB
- Summary size including object overhead
- Compression ratio vs. dense matrix

API: [`AdaptiveCrossApproximation.storage`](@ref)

## Operations

HMatrix integrates with Julia's LinearAlgebra interface:

```julia
# Matrix-vector product
y = hmat * x

# Transpose and adjoint
y = transpose(hmat) * x
y = adjoint(hmat) * x

# Conversion to dense
A_dense = Matrix(hmat)

# Matrix dimensions
m, n = size(hmat)
```



## Integration with Other Components

HMatrix assembly integrates with:

- **Compression**: [`ACA`](@ref), `acaᵀ` for far-field blocks
- **Pivoting**: Geometric strategies (leja, fill distance), random approaches (randomsampling) or value-based (maximum value)
- **Convergence**: Frobenius norm estimators or randomsampling-based error estimation
- **Trees**: H2Trees.jl or custom hierarchical tree implementations
- **Parallelization**: OhMyThreads schedulers for level-wise assembly

## References

1. **Original H-Matrix theory**: Hackbusch, W. (1999). *A Sparse Matrix Arithmetic Based on H-Matrices*
2. **Hierarchical approximations**: Börm, S., Grasedyck, L., & Hackbusch, W. (2003). *Introduction to Hierarchical Matrices with Applications*
3. **CaCa method**: Börm, S. & Christophersen, S. (2015). *Approximating the matrix exponential via CaCa*
4. **Nested Cross Approximation**: Harbrecht, H. & Schneider, R. (2022). *Nested Cross Approximation*
