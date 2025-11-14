# Adaptive Cross Approximation

## Introduction

The Adaptive Cross Approximation (ACA) algorithm computes a low-rank approximation of a matrix $ \bm A^{m \times n}$ using only a small subset of rows and columns. 
The algorithm builds a factorization of the form

$$\bm A \approx \bm U \bm V^T = \sum_{k=1}^r \bm u_k \bm v_k^T$$

where $\bm U \in \mathbb{R}^{m \times r}$ and $\bm V \in \mathbb{R}^{n \times r}$ are computed iteratively by selecting rows and columns of $\bm A$.

## Algorithm

The ACA algorithm proceeds as follows:

1. **Select and sample first row**: $\bm v_1^\text{T} = \bm A[i_1, :]$
2. **Select column**: Choose column index $j_1$ 
3. **Sample column**: Sample column $\bm u_1 = A[:, j_1]$
4. **Normalize**: $\bm v_1 = \bm v_1 / \bm v_{1, j_1}$
5. **Iterate**: Until convergence criterion is met, for $r = 2, 3, \ldots$:
   - Select row index $i_r$ 
   - Sample and update row: $\bm v_r^T = \bm A[i_r, :] - \sum_{k=1}^{r-1} \bm u_{k, i_r} \bm v_k^T$
   - Select column index $j_r$ 
   - Sample and update column: $\bm u_r = \bm A[i_r, :] -\sum_{k=1}^{r-1} \bm u_k \bm v_{k, j_r}$
   - Normalize: $\bm v_r =\bm v_r / \bm v_{r, j_r}$

API: [`ACA`](@ref)

**ACAᵀ**:
The column-first variant starts by selecting a column, then a row, reversing the standard order. 
This can be advantageous when the matrix structure favors column operations.

API: [`AdaptiveCrossApproximation.ACAᵀ`](@ref) 

### Pivoting
To select the row and column indices $i_r$ and $j_r$ different pivoting strategies can be employed. 
In this package several strategies are implemented, e.g., maximum value pivoting, random sampling pivoting, fill distance pivoting, Leja2 pivoting, mimicry pivoting, and tree mimicry pivoting. 
For more details see the [pivoting strategy](../details/pivoting.md) documentation.

### Convergence Criteria
To determine when to stop the iteration different convergence criteria can be used.
In this package several criteria are implemented, e.g., the Frobenius norm estimator, a random sample based criterion, and an extrapolation based criterion. 
For more details see the [convergence criteria](../details/convergence.md) documentation.


