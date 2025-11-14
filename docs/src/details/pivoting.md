# Pivoting Strategies

Pivoting strategies determine how rows and columns are selected during ACA compression. The choice of pivoting strategy significantly affects both the accuracy and computational cost of the approximation.

## Value-Based Strategies

Value-based strategies select pivots by examining matrix entries to find the most significant components.

### Maximum Value Pivoting

The maximum value strategy, also referred to as partial pivoting [[1, 2]](@ref refs), selects the pivot with the largest absolute value in the current residual. 
In the standard ACA when starting with the first row, all following columns are selected as

$$\argmax_{j} |\bm v_{r, j}|$$

and all rows for $r > 1$ are selected as

$$\argmax_{i} |\bm u_{r-1, i}|\,.$$

**API:** [`MaximumValue`](@ref)

### Random Sampling

Random sampling pivoting is typically combined with the random sampling convergence criterion or a combined convergence criterion.
It selects the next row or column leveraging the randomly sampled entries of the underlying matrix used in the convergence criterion, choosing row or column of the sample with the maximum absolute error after the $r$-th iteration, following 

$$\argmax_{k} |\bm e_{r, k}|\,, $$

where $\bm e_r$ contains the error of the sampled entries after the $r$-th iteration.

In the random sampling convergence criterion, the mean error of the random samples is used to estimate the overall residual error.


API: [`RandomSamplingPivoting`](@ref)

## Geometry-Based Strategies

Geometry-based strategies exploit spatial information about the underlying point sets or basis functions.

### Fill Distance
The fill distance strategy, following [[3]](@ref refs) selects the row or column associated with geometrical positions $\bm x \in X$ that minimize the fill distance

$$\bm h\coloneqq \sup_{\bm x \in X}\text{dist}(\bm x, X_r)\,,$$

where $\text{dist}(\bm x, X_r) = \min_{\bm y \in X_r} |\bm x - \bm y|$ and $X_r$ is the set of already selected points associated with rows or columns after $r$ iterations, from one step to the next.
This strategy aims to cover the domain uniformly, ensuring that no region is left unrepresented.

*Note: this strategy should be used only either for the rows or the columns, not both simultaneously and be combined with partial pivoting.*


API: [`FillDistance`](@ref)

### Modified Leja Points
Modified Leja points, following [[5]](@ref refs), follow a similar idea to the fill distance strategy but instead of minimizing the fill distance in each iteration selects the node furthest away from the already selected points $X_r$:

$$\argmax_i (\bm h_i)$$

This approach results in a similar geometrical distribution as the fill distance strategy, however, it is significantly more efficient.

*Note: this strategy should be used only either for the rows or the columns, not both simultaneously and be combined with partial pivoting.*


API: [`Leja2`](@ref)

### Mimicry Pivoting
Mimicry pivoting, following [[7]](@ref refs), selects rows or columns based on geometric information to mimic the geometric distribution of the positions associated with the rows or columns picked by the partial pivoting. 
This is achieved by solving in each the maximization problem

$$\argmax_{j:\bm x_j \in X} \biggr[ \big(\text{dist}(\bm x_j, X_r)\big)\biggr(\prod_{\bm x_i \in X_r}|\bm x_i-\bm x_j|\biggr)^{1/r}\big(|\bm x_j-\bm c|\big)^{-4}\biggr]\,,$$

where $\bm c$ is the geometric center of the positions associated with the rows when columns are selected and the other way around.

This strategy is designed for the incomplete ACA (iACA).


*Note: this strategy should be used only either for the rows or the columns, not both simultaneously and be combined with partial pivoting.*

API: [`MimicryPivoting`](@ref)

### Tree Mimicry Pivoting
Tree mimicry pivoting extends the mimicry pivoting strategy by leveraging a hierarchical clustering of the geometric positions associated with the rows and columns. 
The hierarchical clustering of the positions hast to be passed to the pivoting strategy.
For details see [[7]](@ref refs).

API: [`TreeMimicryPivoting`](@ref)

## Combined Strategies

The combined pivoting strategy allows mixing different pivoting strategies combined with multiple convergence criteria, that decide which strategy to use at each step, enabling hybrid approaches.

API: [`AdaptiveCrossApproximation.CombinedPivStrat`](@ref)

## Choosing a Strategy
The choice of pivoting strategy should be guided by the specific characteristics of the problem at hand, including the nature of the matrix, desired accuracy, and computational resources.
In general the solid results are obtained using the maximum value pivoting strategy.

For spatial problems where the zeros blocks arise in the matrix structure, geometry-based strategies or random sampling pivoting can provide better performance. For details see [[6]](@ref refs).
