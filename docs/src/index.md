# AdaptiveCrossApproximation

This package provides implementations of adaptive cross approximation and several
of its variants [[1, 3, 6]](@ref refs). It supports different pivoting strategies and
convergence criteria [[2, 4, 5]](@ref refs). The package also contains the
incomplete adaptive cross approximation for efficient pivot selection in the
construction of $\mathcal{H}^2$-matrices [[7]](@ref refs).

The package also assembles $\mathcal{H}$-matrices for point kernels and BEAST
operators and provides their arithmetic: sums, products, recompression, and an
LU factorization for direct solves and preconditioning [[8, 9]](@ref refs). For transpose-symmetric operators, single-space assembly shares
off-diagonal near blocks and far-field factors. See
[Symmetric Assembly on One Space](@ref symmetric-hmatrix-assembly).

## Installation 
Installing AdaptiveCrossApproximation is done by entering the package manager (enter `]` at the julia REPL) and issuing:

```
pkg> add https://github.com/JoshuaTetzner/AdaptiveCrossApproximation.jl.git
```


## [References](@id refs)

- [1] Bebendorf, M., and S. Rjasanow. “Adaptive Low-Rank Approximation of Collocation Matrices.” *Computing* 70, no. 1 (2003): 1–24. [https://doi.org/10.1007/s00607-002-1469-6](https://doi.org/10.1007/s00607-002-1469-6).
- [2] De Marchi, Stefano. “On Leja Sequences: Some Results and Applications.” *Applied Mathematics and Computation* 152, no. 3 (2004): 621–47. [https://doi.org/10.1016/S0096-3003(03)00580-0](https://doi.org/10.1016/S0096-3003%2803%2900580-0).
- [3] Zhao, K., M. N. Vouvakis, and J. F. Lee. “The Adaptive Cross Approximation Algorithm for Accelerated Method of Moments Computations of EMC Problems.” *IEEE Transactions on Electromagnetic Compatibility* 47, no. 4 (2005): 763–73. [https://doi.org/10.1109/TEMC.2005.857898](https://doi.org/10.1109/TEMC.2005.857898).
- [4] Heldring, Alexander, Eduard Ubeda, and Juan M. Rius. “Improving the Accuracy of the Adaptive Cross Approximation with a Convergence Criterion Based on Random Sampling.” *IEEE Transactions on Antennas and Propagation* 69, no. 1 (2021): 347–55. [https://doi.org/10.1109/TAP.2020.3010857](https://doi.org/10.1109/TAP.2020.3010857).
- [5] Bauer, M., M. Bebendorf, and B. Feist. “Kernel-Independent Adaptive Construction of $\mathcal{H}^2$-Matrix Approximations.” *Numerische Mathematik* 150, no. 1 (2022): 1–32. [https://doi.org/10.1007/s00211-021-01255-y](https://doi.org/10.1007/s00211-021-01255-y).
- [6] Tetzner, Joshua M., and Simon B. Adrian. “On the Adaptive Cross Approximation for the Magnetic Field Integral Equation.” *IEEE Transactions on Antennas and Propagation* 72, no. 12 (2024): 9366–77. [https://doi.org/10.1109/TAP.2024.3483296](https://doi.org/10.1109/TAP.2024.3483296).
- [7] Tetzner, Joshua M., and Simon B. Adrian. “The Incomplete Adaptive Cross Approximation for the Fast Construction of $\mathcal{H}^2$-Matrices and Its Application to the Electric Field Integral Equation for Electrically Small Problems.” *IEEE Transactions on Antennas and Propagation* 74, no. 8 (2026): 7835–47. [https://doi.org/10.1109/TAP.2026.3687274](https://doi.org/10.1109/TAP.2026.3687274).
- [8] Hackbusch, W. “A Sparse Matrix Arithmetic Based on $\mathcal{H}$-Matrices. Part I: Introduction to $\mathcal{H}$-Matrices.” *Computing* 62, no. 2 (1999): 89–108. [https://doi.org/10.1007/s006070050015](https://doi.org/10.1007/s006070050015).
- [9] Grasedyck, L., and W. Hackbusch. “Construction and Arithmetics of $\mathcal{H}$-Matrices.” *Computing* 70, no. 4 (2003): 295–334. [https://doi.org/10.1007/s00607-003-0019-1](https://doi.org/10.1007/s00607-003-0019-1).
- [10] Bebendorf, M. *Hierarchical Matrices: A Means to Efficiently Solve Elliptic Boundary Value Problems.* Lecture Notes in Computational Science and Engineering 63. Berlin: Springer, 2008. [https://doi.org/10.1007/978-3-540-77147-0](https://doi.org/10.1007/978-3-540-77147-0).
- [11] Halko, N., P. G. Martinsson, and J. A. Tropp. “Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions.” *SIAM Review* 53, no. 2 (2011): 217–88. [https://doi.org/10.1137/090771806](https://doi.org/10.1137/090771806).
- [12] Kriemann, R. “$\mathcal{H}$-LU Factorization on Many-Core Systems.” *Computing and Visualization in Science* 16, no. 3 (2013): 105–17. [https://doi.org/10.1007/s00791-014-0226-7](https://doi.org/10.1007/s00791-014-0226-7).
