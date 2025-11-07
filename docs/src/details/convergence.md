# Convergence Criteria

Convergence criteria determine when to stop the ACA iteration, balancing approximation accuracy against computational cost. The choice of convergence criterion affects both the quality and efficiency of the compression.

## Overview

During ACA compression, we build a low-rank approximation iteratively:

$$A \approx UV^T = \sum_{k=1}^r u_k v_k^T$$

After each iteration $k$, we must decide whether to continue or terminate. This decision is based on estimating the approximation error:

$$\|A - UV^T\|$$

Since computing the exact error requires accessing the full matrix (defeating the purpose of compression), we rely on computable error estimates.

## Frobenius Norm Estimation

The most common approach estimates the Frobenius norm of the approximation error. The key insight is that we can track:

$$\|UV^T\|_F^2 = \sum_{i=1}^r \sum_{j=1}^r \langle u_i, u_j \rangle \langle v_i, v_j \rangle$$

This can be computed incrementally as new pivots are added, without accessing the original matrix.

### Standard ACA

For standard ACA with full matrix access, we track the squared Frobenius norm:

$$\|UV^T\|_F^2 = \sum_{k=1}^r \|u_k\|^2 \|v_k\|^2 + 2\sum_{i<j} \langle u_i, u_j \rangle \langle v_i, v_j \rangle$$

At iteration $k$, we check if:

$$\|u_k\| \|v_k\| < \text{tol} \cdot \|UV^T\|_F$$

This provides a relative error estimate. The computation is incremental, adding $O(k)$ work per iteration.

### Incomplete ACA (iACA)

For incomplete ACA with geometric pivoting, we cannot compute full inner products. Instead, we use a moving average:

$$\bar{n}_k = \frac{1}{k} \sum_{i=1}^k \|u_i\| \|v_i\|$$

and check if:

$$\|u_k\| \|v_k\| < \text{tol} \cdot \bar{n}_k$$

This simpler criterion requires only current pivot norms, not historical inner products.

## Extrapolation-Based Criteria

Extrapolation methods attempt to predict the asymptotic behavior of the approximation error by fitting the sequence of pivot norms to a model and extrapolating to estimate future decay rates.

The basic idea is to observe that for well-approximable matrices, the pivot contribution $\|u_k\| \|v_k\|$ often decays exponentially or algebraically:

$$\|u_k\| \|v_k\| \sim C \rho^k \quad \text{or} \quad \|u_k\| \|v_k\| \sim C k^{-\alpha}$$

By fitting such a model to recent pivots, we can estimate when the error will drop below the tolerance without computing all pivots.

**Advantages:**
- Can terminate earlier than norm estimation
- Exploits decay structure of the approximation
- Useful for smooth kernel matrices

**Disadvantages:**
- Requires fitting procedure
- May be unreliable for irregular decay patterns
- Additional computational overhead

## Random Sampling

Random sampling convergence criteria periodically sample matrix entries to estimate the actual error. This provides direct error feedback but requires matrix access.

The algorithm:
1. Select random matrix entries $(i,j)$
2. Compute approximation error $|A_{ij} - (UV^T)_{ij}|$
3. Estimate global error from samples
4. Terminate when estimated error < tolerance

**Use cases:**
- When matrix access is acceptable
- Need reliable error estimates
- Other criteria may be unreliable

## Combined Criteria

The `CombinedConvCrit` allows using multiple convergence checks simultaneously:

- Require all criteria to be satisfied (AND logic)
- Terminate when any criterion is met (OR logic)
- Use different criteria for different phases

This enables sophisticated stopping strategies, such as:
- Use norm estimation as primary criterion
- Add extrapolation for early termination
- Include maximum rank as safety cutoff

## Choosing a Criterion

**Use Frobenius norm estimation when:**
- Standard and well-tested behavior is desired
- Working with general matrices
- Incremental cost is acceptable

**Use extrapolation when:**
- Matrix has smooth decay properties
- Early termination is important
- Willing to accept fitting overhead

**Use random sampling when:**
- Need direct error estimates
- Matrix access is cheap
- Other estimates may be unreliable

**Use combined criteria when:**
- Need robust stopping conditions
- Different phases require different checks
- Want safety bounds with optimistic early termination

## Practical Considerations

### Tolerance Selection

The tolerance parameter controls the accuracy-rank tradeoff:
- Smaller tolerance → higher accuracy, larger rank
- Typical values: $10^{-4}$ to $10^{-8}$
- Should match application requirements

### Numerical Stability

Very small tolerances may encounter numerical issues:
- Floating-point precision limits
- Ill-conditioned pivot rows/columns
- Consider using combined criteria with maximum rank cutoff

### Performance

Convergence checking adds overhead:
- Norm estimation: $O(k)$ per iteration
- Extrapolation: $O(k \log k)$ fitting cost
- Random sampling: depends on sample size

For large problems, the overhead is typically negligible compared to matrix access and linear algebra operations.
