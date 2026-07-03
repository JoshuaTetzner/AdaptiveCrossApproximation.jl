"""
    AdaptiveCrossApproximation

Adaptive Cross Approximation (ACA) algorithms for hierarchical low-rank matrix compression.

Provides full-rank and incomplete adaptive cross approximation algorithms optimized for
boundary integral operators and other kernel-based matrices. Key features:

  - **ACA**: Standard adaptive cross approximation selecting pivots by alternating rows/columns
  - **IACA**: Incomplete variant for geometric pivoting in hierarchical matrix construction
  - **HMatrix**: Hierarchical matrix representation combining dense near-field and low-rank far-field blocks
  - **Pivoting strategies**: Maximum value, geometric (Leja, Fill Distance, Mimicry), random sampling
  - **Convergence criteria**: Frobenius norm estimation, random sampling, extrapolation, combined criteria
  - **Kernel matrices**: Support for BEAST boundary integral operators and point-based kernels

# Main API

**Compressors:**

  - [`ACA`](@ref): Standard row-first variant
  - [`ACAᵀ`](@ref): Column-first variant (dual of `ACA`)
  - [`IACA`](@ref): Incomplete variant for geometric pivoting and hierarchical matrices
  - [`aca`](@ref): Convenience function for matrix compression

**Hierarchical matrices:**

  - [`HMatrix`](@ref): Hierarchical matrix type
  - [`AdaptiveCrossApproximation.H.assemble`](@ref): Automatic assembly with tree-based blocking
  - [`nearmatrix`](@ref), [`farmatrix`](@ref): Extract system components

**Pivoting strategies:**

  - [`MaximumValue`](@ref): Partial pivoting (largest absolute value)
  - [`Leja2`](@ref): Modified Leja point selection
  - [`FillDistance`](@ref): Geometric fill distance minimization
  - [`MimicryPivoting`](@ref): Distributional mimicry without full matrix access
  - [`TreeMimicryPivoting`](@ref): Tree-aware variant for H²-matrices

**Convergence criteria:**

  - [`FNormEstimator`](@ref): Frobenius norm-based stopping (ACA and IACA paths)
  - [`FNormExtrapolator`](@ref): Extrapolation-enhanced convergence detection
  - [`RandomSampling`](@ref): Random sampling-based convergence

**Kernel matrices:**

  - [`AbstractKernelMatrix`](@ref): Interface for lazy matrix entry evaluation
  - [`PointMatrix`](@ref): Point cloud kernel matrices
  - [`BEASTKernelMatrix`](@ref): BEAST boundary integral operator matrices

# Example

```julia
using AdaptiveCrossApproximation

# Compress a matrix with default settings
U, V = aca(M; tol=1e-4, maxrank=50)
approx = U * V

# Or build an HMatrix with automatic tree-based blocking
hmat = H.assemble(operator, testspace, trialspace)
y = hmat * x  # Matrix-vector product
```

# See also

  - BEAST.jl for boundary integral operators
  - H2Trees.jl for hierarchical clustering
  - BlockSparseMatrices.jl for sparse block storage
"""
module AdaptiveCrossApproximation

using LinearAlgebra
using Polynomials
using ProgressMeter
using StaticArrays

include("utils.jl")

include("hmatrix/kernelmatrix/abstractkernelmatrix.jl")
include("hmatrix/kernelmatrix/beastkernelmatrix.jl")
include("hmatrix/kernelmatrix/gpubeastkernelmatrix.jl")
include("hmatrix/kernelmatrix/gpumatrixdata.jl")
include("hmatrix/kernelmatrix/pointmatrix.jl")

include("pivoting/abstractpivoting.jl")
include("convergence/abstractconvergence.jl")

include("pivoting/maxvalue.jl")
include("pivoting/lejapoints.jl")
include("pivoting/filldistance.jl")
include("pivoting/mimicrypivoting.jl")
include("pivoting/treemimicrypivoting.jl")

include("convergence/estimation.jl")
include("convergence/oversamp_ifnorm_est.jl")
include("convergence/extrapolation.jl")
include("convergence/randomsampling.jl")
include("convergence/combinedconvcrit.jl")

include("pivoting/treemimicrypivoting2.jl")

include("pivoting/combinedpivstrat.jl")
include("pivoting/randomsampling.jl")

"""
    nextrc!(buf, A, i, j)

Fill `buf` in place with the entries `A[i, j]` (a sampled row/column block).

Interface method used by [`ACA`](@ref), [`ACAᵀ`](@ref) and [`IACA`](@ref) to pull the
row/column samples they pivot on, so `A` need not materialize as a dense matrix.
The fallback here covers any `AbstractArray`; kernel-matrix backends
([`AbstractKernelMatrix`](@ref) subtypes such as `BEASTKernelMatrix` or
[`PointMatrix`](@ref)) provide their own methods that evaluate entries on demand.
"""
nextrc!(buf, A::AbstractArray, i, j) = (buf .= view(A, i, j))

include("aca.jl")
include("acaT.jl")
include("iaca.jl")

if !isdefined(Base, :get_extension) # for julia version < 1.9
    include("../ext/ACAH2Trees/ACAH2Trees.jl")
end

include("hmatrix/abstracthmatrix.jl")

module H
    using ..AdaptiveCrossApproximation:
        HMatrix, _tree, defaulttreebackend, defaultmatrixdata, tofarquadstrat

    """
        H.assemble(op, testspace, trialspace; tree=..., quadstrat=..., kwargs...)

    Assemble a hierarchical matrix with automatic tree-based blocking.

    High-level convenience function that automatically constructs a hierarchical
    clustering tree and assembles the complete HMatrix with near and far-field blocks.
    This is the recommended entry point for most applications.

    # Arguments

      - `op`: Operator/kernel for matrix entry evaluation

      - `testspace`: Test space (row basis/points)

      - `trialspace`: Trial space (column basis/points)

      - `tree`: Hierarchical tree structure. Auto-generated from
        `AdaptiveCrossApproximation.defaulttreebackend()` if not provided: a
        `H2Trees.TwoNTree` when only H2Trees.jl is loaded, or a k-means clustered
        tree when ParallelKMeans.jl is loaded alongside it.

      - `treekwargs`: Keyword arguments forwarded to the automatic tree construction
        (e.g. `minvalues`, `numberofclusters` for the k-means backend); ignored if
        `tree` is given explicitly.

      - `quadstrat`: Quadrature strategy, matching `BEAST.assemble`'s keyword of the
        same name. Defaults like `BEAST.defaultquadstrat(op, testspace, trialspace)`
        for BEAST operators (via the `ACABEAST` extension).

      - `nearquadstrat`: Quadrature strategy for near-field (dense) blocks. Defaults
        to `quadstrat`.

      - `farquadstrat`: Quadrature strategy for far-field (compressed, admissible)
        blocks. Defaults to `tofarquadstrat(quadstrat)`, which strips singular-quadrature
        handling since far blocks are always well-separated.

      - `kwargs...`: Additional parameters passed to [`HMatrix`](@ref):

          + `tol`: Convergence tolerance (default `1e-4`)
          + `maxrank`: Maximum rank for compression (default `40`)
          + `compressor`: ACA or IACA instance (default `ACA(tol=tol)`)
          + `isnear`: Admissibility predicate (default `isnear()`)
          + `spaceordering`: Space ordering strategy (default `PreserveSpaceOrder()`)

    # Returns

      - `HMatrix`: Assembled hierarchical matrix ready for matrix-vector products

    # Notes

    Requires H2Trees.jl to be loaded for automatic tree generation. If providing
    a custom tree, ensure it implements the tree interface expected by HMatrix assembly.

    # Examples

    ```julia
    # With BEAST boundary integral operators
    hmat = H.assemble(op, testspace, trialspace; tol=1e-5, maxrank=100)
    y = hmat * x  # Efficient matrix-vector product
    ```

    # See also

    [`HMatrix`](@ref)
    """
    function assemble(
        op,
        testspace,
        trialspace;
        treekwargs=(;),
        tree=_tree(defaulttreebackend(), testspace, trialspace; treekwargs...),
        quadstrat=defaultmatrixdata(op, testspace, trialspace),
        nearquadstrat=quadstrat,
        farquadstrat=tofarquadstrat(quadstrat),
        kwargs...,
    )
        return HMatrix(
            op,
            testspace,
            trialspace,
            tree;
            nearmatrixdata=nearquadstrat,
            farmatrixdata=farquadstrat,
            kwargs...,
        )
    end
end

export H
export HMatrix
export ACA
export ACAᵀ
export IACA
export FNormEstimator, FNormExtrapolator, OversampIFNormEst
export MaximumValue, Leja2, FillDistance
export MimicryPivoting, TreeMimicryPivoting
export reset!
export AbstractKernelMatrix, GPUMatrixData
end
