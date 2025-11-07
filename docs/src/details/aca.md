# Adaptive Cross Approximation

## Introduction

The Adaptive Cross Approximation (ACA) algorithm computes a low-rank approximation of a matrix $A \in \mathbb{R}^{m \times n}$ using only a small subset of its entries. The algorithm builds the approximation:

$$A \approx UV^T = \sum_{k=1}^r u_k v_k^T$$

where $U \in \mathbb{R}^{m \times r}$ and $V \in \mathbb{R}^{n \times r}$ are computed iteratively by selecting rows and columns of $A$.

## Algorithm

### Standard ACA (Row-First)

The standard ACA algorithm proceeds as follows:

1. **Initialize**: Set $k = 1$, select initial column index $j_1$
2. **Extract column**: Sample column $u_1 = A[:, j_1]$
3. **Select row**: Choose row index $i_1$ (typically $i_1 = \arg\max_i |u_{1,i}|$)
4. **Extract row**: Sample row $v_1^T = A[i_1, :]$
5. **Normalize**: Set $u_1 \leftarrow u_1 / v_{1,j_1}$
6. **Iterate**: For $k = 2, 3, \ldots$ until convergence:
   - Select column $j_k$ from residual row $v_{k-1}^T$
   - Extract and deflate column: $\tilde{u}_k = A[:, j_k] - \sum_{\ell=1}^{k-1} u_\ell v_{\ell, j_k}$
   - Select row index $i_k$ from $\tilde{u}_k$
   - Extract and deflate row: $\tilde{v}_k^T = A[i_k, :] - \sum_{\ell=1}^{k-1} u_{\ell, i_k} v_\ell^T$
   - Normalize: $u_k = \tilde{u}_k / \tilde{v}_{k, j_k}$, $v_k = \tilde{v}_k$
   - Check convergence

### Column-First ACA (ACAᵀ)

The column-first variant starts by selecting a column, then a row, reversing the standard order. This can be advantageous when the matrix structure favors column operations.

## Key Properties

### Partial Pivoting

ACA uses partial pivoting: when selecting a row index $i_k$, only the current column $u_k$ is examined (and vice versa for columns). This makes the algorithm efficient but means it doesn't achieve full pivoting optimality.

### Computational Cost

For an $m \times n$ matrix compressed to rank $r$:
- **Matrix accesses**: $r$ full rows + $r$ full columns = $O(r(m+n))$ entries
- **Deflation cost**: $O(r^2(m+n))$ operations
- **Total**: $O(r^2(m+n))$ assuming matrix access is $O(1)$

This is much cheaper than SVD which requires $O(mn \min(m,n))$ operations.

### Approximation Quality

For matrices with rapidly decaying singular values, ACA produces near-optimal low-rank approximations. The approximation quality depends on:
- Matrix rank structure
- Pivot selection strategy
- Convergence criterion

### No Matrix Assembly

The key advantage of ACA is that it never requires the full matrix $A$ to be assembled. Only selected rows and columns are needed, making it ideal for:
- Kernel matrices: $A_{ij} = K(x_i, y_j)$ where kernel evaluation is expensive
- Boundary element methods
- Hierarchical matrix compression

## Variants

### Incomplete ACA (iACA)

Incomplete ACA is designed for scenarios where:
- Individual matrix entries cannot be accessed
- Only geometric information is available
- Working with hierarchical matrix structures

Instead of value-based pivoting, iACA uses geometric pivoting strategies that select points based on spatial distribution rather than matrix values. This enables compression of matrix blocks in hierarchical formats without expensive kernel evaluations.

**Key differences:**
- Pivoting based on geometry, not matrix values
- Cannot perform full deflation (no individual entry access)
- Uses simplified convergence criteria
- Particularly efficient with tree-based pivoting strategies

### Fully Pivoted ACA

Fully pivoted ACA would select each pivot from the entire remaining submatrix rather than just the current row/column. While theoretically optimal, this requires $O(mn)$ work per pivot, defeating the purpose of the algorithm.

## Mathematical Foundation

### Connection to Singular Value Decomposition

For a matrix with singular value decomposition $A = \sum_{i=1}^{\min(m,n)} \sigma_i u_i v_i^T$, the optimal rank-$r$ approximation is:

$$A_r^* = \sum_{i=1}^r \sigma_i u_i v_i^T$$

ACA produces an approximation $A_r^{ACA}$ that, under suitable conditions, satisfies:

$$\|A - A_r^{ACA}\|_F \lesssim C \|A - A_r^*\|_F$$

where $C$ is a modest constant depending on the matrix structure.

### Approximability

ACA works well when the matrix has a rapidly decaying singular value spectrum. For matrices arising from smooth kernels, exponential decay is common:

$$\sigma_k \sim e^{-\alpha k}$$

In such cases, ACA with tolerance $\varepsilon$ typically requires rank:

$$r \approx \frac{1}{\alpha} \log(1/\varepsilon)$$

## Applications

### Kernel Matrices

For kernel matrices $A_{ij} = K(x_i, y_j)$ arising from smooth kernels on separated point clusters:
- ACA avoids computing all $O(mn)$ kernel evaluations
- Only $O(r(m+n))$ evaluations needed
- Combined with hierarchical decomposition enables fast methods

### Boundary Element Methods

In BEM, interaction matrices between well-separated surface patches:
- Have low-rank structure
- Kernel evaluations are expensive (require numerical integration)
- ACA enables efficient assembly and matrix-vector products

### Hierarchical Matrices

ACA serves as the compression engine in hierarchical matrix formats:
- Compress far-field blocks in $\mathcal{H}$-matrices
- Build $\mathcal{H}^2$-matrix representations
- Enable linear complexity operations for elliptic PDEs