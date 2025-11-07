<p align="center">
<picture>
  <source media="(prefers-color-scheme)" srcset="docs/src/assets/logoREADME.svg" height="90">
  <img alt="" src="" height="80">
</picture>
</p>

[![Docs-stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JoshuaTetzner.github.io/AdaptiveCrossApproximation.jl/stable/)
[![Docs-dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JoshuaTetzner.github.io/AdaptiveCrossApproximation.jl/dev/)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/JoshuaTetzner/AdaptiveCrossApproximation.jl/blob/master/LICENSE)
[![CI](https://github.com/JoshuaTetzner/AdaptiveCrossApproximation.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/JoshuaTetzner/AdaptiveCrossApproximation.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/JoshuaTetzner/AdaptiveCrossApproximation.jl/graph/badge.svg?token=RDRQTBWQS3)](https://codecov.io/gh/JoshuaTetzner/AdaptiveCrossApproximation.jl)

## Introduction

This package provides different flavors of the adaptive cross approximation introduced in BEBENDORF.
Beside the standard algorithm this package allows to use several different pivoting strategies [] and convergence criteria[].
Further more this package contains an incomplete adaptive cross approximation allowing an efficient pivoting selection for the construction of $\mathcal{H}^2$-matrices [].

## Installation 
Installing AdaptiveCrossApproximation is done by entering the package manager (enter `]` at the julia REPL) and issuing:

```
pkg> add https://github.com/JoshuaTetzner/AdaptiveCrossApproximation.jl.git
```

## [References](@id refs)
The implementation is based on
- [1] Bebendorf, Mario. *Adaptive Cross Approximation of Multidimensional Arrays*. Computing 70, no. 1 (January 2003): 1–24. [https://doi.org/10.1007/s00607-002-0019-1](https://doi.org/10.1007/s00607-002-0019-1).
- [2] Bauer, M., M. Bebendorf, and B. Feist. *Kernel-Independent Adaptive Construction of H²-Matrix Approximations.* Numerische Mathematik 150, no. 1 (January 2022): 1–32. [https://doi.org/10.1007/s00211-021-01255-y](https://doi.org/10.1007/s00211-021-01255-y).
- [3] Heldring, Alexander, Eduard Ubeda, and Juan M. Rius. *Improving the Accuracy of the Adaptive Cross Approximation with a Convergence Criterion Based on Random Sampling.* IEEE Transactions on Antennas and Propagation 69, no. 1 (January 2021): 347–55. [https://doi.org/10.1109/TAP.2020.3010857](https://doi.org/10.1109/TAP.2020.3010857).
- [4] De Marchi, Stefano. *On Leja Sequences: Some Results and Applications.* Applied Mathematics and Computation 152, no. 3 (2004): 621–47. [https://doi.org/10.1016/S0096-3003(03)00580-0](https://doi.org/10.1016/S0096-3003(03)00580-0).
- [5] Tetzner, Joshua M., and Simon B. Adrian. *On the Adaptive Cross Approximation for the Magnetic Field Integral Equation.* Preprint. Preprints, January 26, 2024. [https://doi.org/10.36227/techrxiv.170630205.56494379/v1](https://doi.org/10.36227/techrxiv.170630205.56494379/v1).
