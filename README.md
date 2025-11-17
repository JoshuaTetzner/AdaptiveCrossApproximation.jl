<p align="center">
<picture>
  <source media="(prefers-color-scheme)" srcset="docs/src/assets/logoREADME.svg" height="90">
  <img alt="" src="" height="80">
</picture>
</p>

[![Docs-stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JoshuaTetzner.github.io/AdaptiveCrossApproximation.jl/)
[![Docs-dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JoshuaTetzner.github.io/AdaptiveCrossApproximation.jl/dev/)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/JoshuaTetzner/AdaptiveCrossApproximation.jl/blob/master/LICENSE)
[![CI](https://github.com/JoshuaTetzner/AdaptiveCrossApproximation.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/JoshuaTetzner/AdaptiveCrossApproximation.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/JoshuaTetzner/AdaptiveCrossApproximation.jl/graph/badge.svg?token=RDRQTBWQS3)](https://codecov.io/gh/JoshuaTetzner/AdaptiveCrossApproximation.jl)

## Introduction

This package provides different flavors of the adaptive cross approximation [1, 2].
Beside the standard algorithm this package allows to use several different pivoting strategies and convergence criteria.
Further more this package contains an incomplete adaptive cross approximation allowing an efficient pivoting selection for the construction of $\mathcal{H}^2$-matrices.

## Installation 
Installing AdaptiveCrossApproximation is done by entering the package manager (enter `]` at the julia REPL) and issuing:

```
pkg> add https://github.com/FastBEAST/AdaptiveCrossApproximation.jl.git
```


## References
- [1] Bebendorf, Mario. *Adaptive Cross Approximation of Multidimensional Arrays*. Computing 70, no. 1 (January 2003): 1–24. [https://doi.org/10.1007/s00607-002-0019-1](https://doi.org/10.1007/s00607-002-0019-1).
- [2] Zhao, K., M.N. Vouvakis, and J.-F. Lee. “The Adaptive Cross Approximation Algorithm for Accelerated Method of Moments Computations of EMC Problems.” IEEE Transactions on Electromagnetic Compatibility 47, no. 4 (2005): 763–73. https://doi.org/10.1109/TEMC.2005.857898.
- [3] Bauer, M., M. Bebendorf, and B. Feist. *Kernel-Independent Adaptive Construction of $\mathcal {H}^2$-Matrix Approximations.* Numerische Mathematik 150, no. 1 (January 2022): 1–32. [https://doi.org/10.1007/s00211-021-01255-y](https://doi.org/10.1007/s00211-021-01255-y).
- [4] Heldring, Alexander, Eduard Ubeda, and Juan M. Rius. *Improving the Accuracy of the Adaptive Cross Approximation with a Convergence Criterion Based on Random Sampling.* IEEE Transactions on Antennas and Propagation 69, no. 1 (January 2021): 347–55. [https://doi.org/10.1109/TAP.2020.3010857](https://doi.org/10.1109/TAP.2020.3010857).
- [5] De Marchi, Stefano. *On Leja Sequences: Some Results and Applications.* Applied Mathematics and Computation 152, no. 3 (2004): 621–47. [https://doi.org/10.1016/S0096-3003(03)00580-0](https://doi.org/10.1016/S0096-3003(03)00580-0).
- [6] Tetzner, Joshua M., and Simon B. Adrian. “On the Adaptive Cross Approximation for the Magnetic Field Integral Equation.” IEEE Transactions on Antennas and Propagation, 2024, 1–1. https://doi.org/10.1109/TAP.2024.3483296.
