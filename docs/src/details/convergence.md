# Convergence Criteria

Convergence criteria determine when to stop the ACA iteration.

## Frobenius Norm Estimation

The Frobenius norm estimator is the standard convergence criterion for ACA algorithms [[1, 2]](@ref refs). It estimates the Frobenius norm of the residual matrix as well as the full matrix using only the entries evaluated during the ACA process.

### ACA
For ACA, the squared Frobenius norm of the full matrix is tracked incrementally by

$$\| A^{m\times n}  \|_\text{F}^2 \approx \|\bm U \bm V^T\|_\text{F}^2 = \sum_{k=1}^r |\bm u_k|^2|\bm v_k|^2 + 2\sum_{i<j} \langle \bm u_i, \bm u_j \rangle \langle \bm v_i, \bm v_j \rangle\,.$$

At iteration $r$, the algorithm checks if

$$|\bm u_r| |\bm v_r| < \varepsilon \cdot \|\bm U \bm V^T\|_F\,.$$



API: [`FNormEstimator`](@ref)

### Incomplete ACA (iACA)

For incomplete ACA less entries of the matrix are sampled, therefore, in the case of the the Frobenius norm of the full matrix is estimated following [[7]](@ref refs) by

$$\|A^{m\times n} \|_F^2 \approx \sqrt{\frac{n}{r}} \|A^{m\times r}\|_\text{F}$$

and at iteration $r$ the algorithm checks if

$$|a| < \varepsilon \frac{ \|A^{m\times r}\|_\text{F}}{\sqrt{r}} .$$

This simpler criterion requires only current pivot norms, not historical inner products.

API: [`iFNormEstimator`](@ref)

## Random Sampling

Random sampling convergence criteria estimates the residual error by computing the true error for a set of randomly selected matrix entries, following [[4, 6]](@ref refs), and checks at iteration $r$ if

$$\sqrt{\text{mean}(|\bm e_{r}²|)mn} < \varepsilon \|\bm U \bm V^T\|_\text{F}$$


API: [`AdaptiveCrossApproximation.RandomSampling`](@ref)

## Extrapolation-Based Criteria

Extrapolation criteria enhances the Frobenius norm estimation by predicting the residual norm decay based on previous iterations, following [[7]](@ref refs). 
This criterion can be used to smooth out fluctuations in the estimated error and prevent premature convergence. 
At iteration $r$, if the Frobenius norm estimation is satisfied, the algorithm fits the logarithm of the residual norms of the previous iterations to a quadratic polynomial and extrapolates to the rth iteration and checks if 
$$P²(r) < \log(\varepsilon \|\bm U \bm V^T\|_F)$$

API: [`FNormExtrapolator`](@ref)

## Combined Criteria

The combined convergence criterion allows to combine multiple convergence criteria which all have to be satisfied to terminate the ACA process.
This criterion can be used to control the operating pivoting strategy. 

API: [`AdaptiveCrossApproximation.CombinedConvCrit`](@ref)

## Choosing a Criterion

The choice of convergence criterion should be guided by the specific characteristics of the problem and computational constraints.
The standard Frobenius norm estimation is in the case of the ACA suitable for most applications.
For the IACA the Extrapolation based criterion is recommended.

