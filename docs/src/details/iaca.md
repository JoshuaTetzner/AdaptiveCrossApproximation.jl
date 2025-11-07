# Incomplete Adaptive Cross Approximation

## Motivation

Incomplete ACA (iACA) addresses a fundamental limitation of standard ACA: the requirement to access individual matrix entries. In many applications, particularly hierarchical matrix compression, we need to compress matrix blocks where:

1. **No entry access**: Individual entries $A_{ij}$ cannot be efficiently computed
2. **Geometric information available**: Point positions or spatial structure is known
3. **Batch operations only**: Can extract full rows/columns but not single entries

iACA solves this by using **geometric pivoting strategies** that select pivots based on spatial distribution rather than matrix values.

## The Challenge

Standard ACA requires value-based pivoting to select the next pivot:

$$i_{k+1} = \arg\max_i |r_{ij_k}|$$

where $r_{ij_k}$ is a residual entry. Computing this maximum requires:
- Access to individual residual entries
- Full deflation: $r_{ij} = A_{ij} - \sum_{\ell=1}^k u_{\ell i} v_{\ell j}$

For kernel matrices in hierarchical formats:
- $A_{ij} = K(x_i, y_j)$ requires expensive kernel evaluation
- Deflation requires $O(k)$ evaluations per entry
- Examining all entries for maximum is prohibitive

## Geometric Pivoting

Instead of examining matrix values, iACA selects pivots based on geometric criteria applied to the point sets $X = \{x_1, \ldots, x_m\}$ and $Y = \{y_1, \ldots, y_n\}$.

### Fill Distance Strategy

Select points maximally separated from already chosen points:

$$x_{k+1} = \arg\max_{x \in X \setminus S_k} \min_{y \in S_k} \|x - y\|$$

This ensures good spatial coverage without matrix access.

### Leja Points

Select points to maximize a product criterion:

$$x_{k+1} = \arg\max_{x \in X \setminus S_k} \prod_{i=1}^k \|x - x_i\|$$

Provides excellent distribution properties with theoretical guarantees.

### Mimicry-Based Strategies

The most powerful approach for hierarchical matrices: **reuse pivot patterns** from previous compressions.

**Key insight**: When compressing multiple matrix blocks with similar geometric structure:
1. Compress first block using any geometric strategy
2. Store the selected pivot indices
3. Reuse these indices for subsequent similar blocks
4. Avoid repeated pivot selection computation

This dramatically reduces overhead in hierarchical matrix construction where thousands of similar blocks must be compressed.

## Algorithm Variants

### Row Matrix Variant

For iACA with geometric row pivoting and value-based column pivoting:

1. **Select row geometrically**: $i_k$ based on point positions
2. **Extract row**: $v_k^T = A[i_k, :]$ (full row extraction)
3. **Select column by value**: $j_k = \arg\max_j |v_{kj}|$
4. **Extract and deflate column**: $\tilde{u}_k = A[:, j_k] - \sum_{\ell<k} u_\ell v_{\ell j_k}$
5. Normalize and continue

**Key property**: Only full row/column extractions needed, no individual entry access.

### Column Matrix Variant

Symmetric variant starting with geometric column selection:

1. Select column geometrically
2. Extract full column
3. Select row by maximum value
4. Extract and deflate row
5. Continue

### Pure Geometric Variant

Both rows and columns selected geometrically:
- No matrix access for pivoting
- Most efficient for hierarchical matrices
- Requires good geometric strategies

## Tree-Based Acceleration

### Brute Force Approach

Computing fill distance or Leja criteria naively requires:
- $O(mk)$ distance computations at iteration $k$
- Total $O(mr^2)$ for rank $r$

For large point sets, this becomes expensive even though no matrix access is needed.

### Tree-Based Approach

Using spatial trees (quadtree, octree, k-d tree):

1. **Preprocessing**: Build tree structure $O(m \log m)$
2. **Pivot selection**: Use tree to find next pivot in $O(\log m)$ per iteration
3. **Total cost**: $O(r \log m)$ instead of $O(mr^2)$

The `TreeMimicryPivoting` strategy implements this efficiently, navigating the tree to quickly identify optimal geometric pivots.

**Additional benefits**:
- Natural hierarchical structure for matrix blocks
- Can share tree across multiple compressions
- Enables efficient mimicry by tree traversal patterns

## Convergence Criteria

Standard convergence criteria require inner products:

$$\|UV^T\|_F^2 = \sum_{i,j} \langle u_i, u_j \rangle \langle v_i, v_j \rangle$$

iACA cannot compute these without entry access. Instead, use **simplified criteria**:

### Moving Average Norm

Track the moving average:

$$\bar{n}_k = \frac{1}{k} \sum_{i=1}^k \|u_i\| \|v_i\|$$

Terminate when:

$$\|u_k\| \|v_k\| < \text{tol} \cdot \bar{n}_k$$

This requires only row/column norms, not full inner products.

### Fixed Rank

Simply compress to a predetermined rank based on a priori estimates or hierarchical matrix theory.

## Performance Characteristics

### Computational Cost

For rank-$r$ compression of $m \times n$ matrix:
- **Matrix access**: $r$ rows + $r$ columns = $O(r(m+n))$ entries
- **Pivot selection**: $O(r \log m)$ with trees vs $O(mr^2)$ brute force
- **Deflation**: $O(r^2(m+n))$ same as standard ACA

### Memory

- **Point coordinates**: $O(m+n)$ spatial data
- **Tree structure**: $O(m+n)$ for spatial tree
- **Buffers**: Same as standard ACA

### Accuracy

Geometric pivoting generally achieves similar accuracy to value-based pivoting when:
- Kernel is smooth (exponentially decaying in distance)
- Point sets are well-distributed
- Separation ratio is favorable

For rough kernels or poorly distributed points, value-based pivoting may be superior.

## Use Cases

**Use iACA when:**
- Working with hierarchical matrix formats ($\mathcal{H}$, $\mathcal{H}^2$)
- Individual kernel evaluations are expensive
- Geometric information is naturally available
- Many similar blocks need compression (use mimicry)

**Use standard ACA when:**
- Matrix entries are cheap to access
- No geometric structure available
- Maximum accuracy is critical
- Working with single matrix compression

## Integration with Hierarchical Matrices

iACA is the natural compression method for hierarchical matrices:

1. **Block clustering**: Partition points into spatial clusters
2. **Admissibility**: Identify far-field block pairs
3. **Compression**: Use iACA with geometric pivoting for each admissible block
4. **Mimicry**: Share pivot patterns among geometrically similar blocks
5. **Tree structure**: Use same spatial tree for all operations

This workflow enables:
- $O(n \log n)$ or $O(n)$ storage for $n \times n$ matrices
- $O(n \log n)$ or $O(n)$ matrix-vector products
- Efficient assembly without forming full matrix
