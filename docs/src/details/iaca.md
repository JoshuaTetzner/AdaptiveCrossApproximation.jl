# Incomplete Adaptive Cross Approximation (IACA)

## Introduction

The Incomplete Adaptive Cross Approximation (IACA) is a variant of the Adaptive Cross Approximation (ACA) designed for the efficient construction of nested low-rank representations within the Nested Cross Approximation (NCA) framework.

In contrast to the standard ACA, which computes full rows and columns for pivot selection, the IACA restricts computations to only those matrix elements required for the final nested representation. This reduces unnecessary evaluations and enables a more efficient construction of H2-matrices.

The IACA constructs a factorization of the form

```math
\mathbf{A} \approx \mathbf{U}\mathbf{V}^T = \sum_{k=1}^r \mathbf{u}_k \mathbf{v}_k^T
```

but only evaluates a subset of entries corresponding to selected pivot indices. This makes it particularly suitable for hierarchical methods such as the NCA.

API: [`iACA`](@ref)

---

## Algorithm

The IACA algorithm computes a low-rank approximation of a matrix `A^{m×n}` using only a subset of rows and columns that are required for the nested representation.
The algorithm builds a factorization of the form

```math
\mathbf{A} \approx \mathbf{U}\mathbf{V}^T = \sum_{k=1}^r \mathbf{u}_k \mathbf{v}_k^T
```

where `U ∈ ℝ^{m×r}` and `V ∈ ℝ^{n×r}` are computed iteratively.

The IACA algorithm proceeds as follows:

1. **Select and sample first column**: `u₁ = A[:, j₁]`, where `j₁` is selected using mimicry pivoting.
2. **Select row**: Choose row index `i₁ = argmax |u₁|`.
3. **Initialize**: Set `v₁` implicitly via normalization.
4. **Iterate**: Until convergence criterion is met, for `r = 2, 3, ...`:
   - Select column index `j_r` using mimicry pivoting.
   - Sample and update column:
     ```math
     \mathbf{u}_r = \mathbf{A}[:, j_r] - \sum_{k=1}^{r-1} \mathbf{u}_k \, \mathbf{v}_{k,j_r}
     ```
   - Select row index `i_r = argmax |u_r|`.
   - Update factor entries incrementally using previously computed columns  
   - Normalize implicitly via the pivot entry  

5. **Stop** when the convergence criterion is satisfied


### Pivoting

Instead of selecting pivots via maximum residual entries (as in ACA), the IACA uses a **geometric heuristic** that mimics the spatial distribution of ACA pivots.
This approach we call **mimicry pivoting**, for details see [`MimicryPivoting`](@ref).

### Convergence Criterion

Since the full residual is not available, the classical ACA stopping criterion cannot be used.

Instead, the IACA estimates the residual using the [`iFNormEstimator`](@ref) which provides an estimate of the Frobenius norm of the residual using only rows or columns.

To avoid premature convergence due to inaccurate estimates:

- A **trend-based check** is applied
- A polynomial fit to previous residual estimates is used
- Iteration continues if the predicted residual does not follow the expected decay

This approach is available as the [`FNormExtrapolator`](@ref).

---

## Implementation Notes

- Requires access to **geometric information** (e.g., basis function positions)
- Can be combined with:
  - Representor set strategies
  - Tree-based clustering
- A **tree-based variant (tree mimicry pivoting)** improves scalability
