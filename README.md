<p align="center">
<picture>
  <source media="(prefers-color-scheme: light)" srcset="docs/src/assets/logoREADME.svg" height="90">
  <source media="(prefers-color-scheme: dark)" srcset="docs/src/assets/logo_darkmode.svg" height="90">
  <img alt="" src="" height="80">
</picture>
</p>

[![Docs-stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JoshuaTetzner.github.io/AdaptiveCrossApproximation.jl/stable/)
[![Docs-dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JoshuaTetzner.github.io/AdaptiveCrossApproximation.jl/dev/)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/JoshuaTetzner/AdaptiveCrossApproximation.jl/blob/main/LICENSE)
[![CI](https://github.com/JoshuaTetzner/AdaptiveCrossApproximation.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JoshuaTetzner/AdaptiveCrossApproximation.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/JoshuaTetzner/AdaptiveCrossApproximation.jl/graph/badge.svg?token=RDRQTBWQS3)](https://codecov.io/gh/JoshuaTetzner/AdaptiveCrossApproximation.jl)

## Introduction

This package provides implementations of adaptive cross approximation and several
of its variants [1, 3, 6]. It supports different pivoting strategies and convergence
criteria [2, 4, 5]. The package also contains the incomplete adaptive cross
approximation for efficient pivot selection in the construction of
$\mathcal{H}^2$-matrices [7].

## Features

The following capabilities are implemented (✔) and planned (⌛):

### Low-Rank Approximation

- ✔ Adaptive Cross Approximation (ACA) and its column-first transpose dual ACAᵀ
- ✔ Full, partial, geometric, and random-sampling pivoting strategies
- ✔ Standard, random-sampling, combined, and extrapolated convergence criteria
- ✔ Incomplete Adaptive Cross Approximation (IACA) for nested low-rank constructions

### Hierarchical Matrices

- ✔ $\mathcal H$-matrices for point kernels and BEAST integral operators
- ✔ Clustering based on H2Trees.jl with geometry-adapted k-means clustering
- ✔ Multithreaded near-field assembly and ACA-compressed far-field assembly
- ✔ Matrix-vector, transpose, and adjoint products through the Julia matrix interface
- ⌛ GPU support for hierarchical assembly and matrix operations
- ⌛ $\mathcal H$-matrix algebra, enabling hierarchical direct solvers

## Installation

Enter the Julia package manager by pressing `]` at the REPL and run:

```
pkg> add https://github.com/JoshuaTetzner/AdaptiveCrossApproximation.jl.git
```

## H-Matrices with BEAST

The convenience `assemble` interface constructs the cluster tree, assembles dense
near-field interactions, and compresses admissible far-field blocks. Loading
ParallelKMeans selects the recommended k-means tree backend automatically.

```julia
using AdaptiveCrossApproximation
using BEAST
using CompScienceMeshes
using H2Trees
using ParallelKMeans

mesh = meshsphere(1.0, 0.3)
space = raviartthomas(mesh)
operator = Maxwell3D.singlelayer(; wavenumber=1.0)

hmat = AdaptiveCrossApproximation.assemble(
    operator,
    space,
    space;
    tol=1e-4,
    maxrank=40,
)

x = ones(eltype(hmat), size(hmat, 2))
y = hmat * x
```

The result behaves as a matrix-free linear operator while retaining direct access
to its near- and far-field representations. See the
[H-matrix manual](https://JoshuaTetzner.github.io/AdaptiveCrossApproximation.jl/dev/manual/hmatrix/)
for configuration options and point-kernel examples.

## Related Packages
- [BEAST.jl](https://github.com/krcools/BEAST.jl) boundary element toolkit in Julia
- [H2Trees.jl](https://github.com/djukic14/H2Trees.jl) hierarchical tree construction for $\mathcal{H}$- and $\mathcal{H}^2$-matrices
- [NestedCrossApproximation.jl](https://github.com/JoshuaTetzner/NestedCrossApproximation.jl) nested low-rank approximation for $\mathcal{H}^2$-matrices
- [BlockSparseMatrices.jl](https://github.com/djukic14/BlockSparseMatrices.jl)
  stores and applies the dense near field.

## References

- [1] Bebendorf, M., and S. Rjasanow. “Adaptive Low-Rank Approximation of Collocation Matrices.” *Computing* 70, no. 1 (2003): 1–24. [https://doi.org/10.1007/s00607-002-1469-6](https://doi.org/10.1007/s00607-002-1469-6).
- [2] De Marchi, Stefano. “On Leja Sequences: Some Results and Applications.” *Applied Mathematics and Computation* 152, no. 3 (2004): 621–47. [https://doi.org/10.1016/S0096-3003(03)00580-0](https://doi.org/10.1016/S0096-3003%2803%2900580-0).
- [3] Zhao, K., M. N. Vouvakis, and J. F. Lee. “The Adaptive Cross Approximation Algorithm for Accelerated Method of Moments Computations of EMC Problems.” *IEEE Transactions on Electromagnetic Compatibility* 47, no. 4 (2005): 763–73. [https://doi.org/10.1109/TEMC.2005.857898](https://doi.org/10.1109/TEMC.2005.857898).
- [4] Heldring, Alexander, Eduard Ubeda, and Juan M. Rius. “Improving the Accuracy of the Adaptive Cross Approximation with a Convergence Criterion Based on Random Sampling.” *IEEE Transactions on Antennas and Propagation* 69, no. 1 (2021): 347–55. [https://doi.org/10.1109/TAP.2020.3010857](https://doi.org/10.1109/TAP.2020.3010857).
- [5] Bauer, M., M. Bebendorf, and B. Feist. “Kernel-Independent Adaptive Construction of $\mathcal{H}^2$-Matrix Approximations.” *Numerische Mathematik* 150, no. 1 (2022): 1–32. [https://doi.org/10.1007/s00211-021-01255-y](https://doi.org/10.1007/s00211-021-01255-y).
- [6] Tetzner, Joshua M., and Simon B. Adrian. “On the Adaptive Cross Approximation for the Magnetic Field Integral Equation.” *IEEE Transactions on Antennas and Propagation* 72, no. 12 (2024): 9366–77. [https://doi.org/10.1109/TAP.2024.3483296](https://doi.org/10.1109/TAP.2024.3483296).
- [7] Tetzner, Joshua M., and Simon B. Adrian. “The Incomplete Adaptive Cross Approximation for the Fast Construction of $\mathcal{H}^2$-Matrices and Its Application to the Electric Field Integral Equation for Electrically Small Problems.” *IEEE Transactions on Antennas and Propagation* 74, no. 8 (2026): 7835–47. [https://doi.org/10.1109/TAP.2026.3687274](https://doi.org/10.1109/TAP.2026.3687274).
