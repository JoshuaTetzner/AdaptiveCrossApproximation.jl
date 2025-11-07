# Pivoting Strategies

Pivoting strategies determine how rows and columns are selected during ACA compression. The choice of pivoting strategy significantly affects both the accuracy and computational cost of the approximation.

## Overview

The Adaptive Cross Approximation algorithm requires selecting a sequence of row and column indices (pivots) to build the low-rank approximation:

$$A \approx \sum_{k=1}^r u_k v_k^T$$

where $u_k$ is a column of $A$ and $v_k$ is a row of $A$. The quality of the approximation depends critically on the pivot selection strategy.

## Value-Based Strategies

Value-based strategies select pivots by examining matrix entries to find the most significant components.

### Maximum Value Pivoting

The maximum value strategy selects the pivot with the largest absolute value in the current residual. For row selection with a known column $j$:

$$i_k = \arg\max_i |A_{ij} - \sum_{\ell=1}^{k-1} u_{\ell,i} v_{\ell,j}|$$

This greedy approach often yields good approximations but requires access to individual matrix entries, which may be expensive for kernel matrices or hierarchical applications.

**Advantages:**
- Simple and intuitive
- Often produces good approximations
- Well-established in the literature

**Disadvantages:**
- Requires full access to matrix entries
- May be expensive for kernel evaluations
- Cannot exploit geometric structure

### Random Sampling

Random sampling selects pivots uniformly at random from the remaining indices. While less accurate than maximum value pivoting, it provides several benefits:

- No matrix access needed for pivot selection
- Can be combined with other strategies
- Useful for stochastic algorithms

## Geometry-Based Strategies

Geometry-based strategies exploit spatial information about the underlying point sets, making them ideal for kernel matrices and hierarchical matrix compression where geometric structure is available.

### Fill Distance

The fill distance strategy selects points that are maximally separated from already selected points. Given positions $X = \{x_1, \ldots, x_n\}$ and already selected points $S_k$, the next point is:

$$x_{k+1} = \arg\max_{x \in X \setminus S_k} \min_{y \in S_k} \|x - y\|$$

This greedy algorithm approximates an optimal coverage of the point set.

**Properties:**
- No matrix access required
- Provides good geometric coverage
- Natural for kernel matrices where $A_{ij} = K(x_i, y_j)$

### Leja Points

Leja points are selected to maximize a certain product criterion. Starting from an initial point $x_1$, subsequent points are chosen as:

$$x_{k+1} = \arg\max_{x \in X \setminus S_k} \prod_{i=1}^k \|x - x_i\|$$

This strategy produces well-distributed points that are particularly effective for interpolation problems.

**Advantages:**
- Strong theoretical foundation
- Excellent distribution properties
- No matrix evaluations needed

### Mimicry Pivoting

Mimicry pivoting reuses pivot patterns from previous compressions, making it highly efficient for hierarchical matrix structures. When compressing similar matrix blocks, the algorithm:

1. Stores pivot sequences from completed compressions
2. Reuses these sequences for new blocks with similar structure
3. Avoids redundant pivot selection computations

This is particularly powerful for hierarchical matrices where many blocks share similar geometric properties.

### Tree Mimicry Pivoting

Tree mimicry pivoting extends the mimicry concept by incorporating hierarchical tree structures. It navigates through a spatial tree (quadtree, octree) to efficiently find and reuse pivot patterns, making it especially efficient for hierarchical matrix compression in multiple dimensions.

## Combined Strategies

The `CombinedPivStrat` allows mixing different pivoting strategies, enabling hybrid approaches such as:

- Start with geometric pivoting for initial coverage
- Switch to value-based pivoting for refinement
- Use random sampling to break ties or add robustness

## Choosing a Strategy

**Use value-based strategies when:**
- Matrix entries are cheap to compute
- Maximum accuracy is required
- Geometric information is unavailable

**Use geometry-based strategies when:**
- Matrix entries are expensive (kernel evaluations)
- Working with hierarchical matrices
- Geometric structure is naturally available
- Need to avoid redundant computations

**Use combined strategies when:**
- Different phases of compression need different approaches
- Want to balance accuracy and efficiency
- Exploiting multiple sources of information
