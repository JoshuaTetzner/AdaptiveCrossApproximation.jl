"""
    PivStrat

Abstract base type for pivoting strategies in cross approximation algorithms.

Pivoting strategies determine which row or column to select at each iteration of the
ACA algorithm. Concrete subtypes are typically stateless and callable, creating
stateful [`PivStratFunctor`](@ref) instances when invoked with index information.

# Subtypes by Category

  - [`GeoPivStrat`](@ref): Geometric strategies (e.g., fill distance, Leja points)
  - [`ValuePivStrat`](@ref): Value-based strategies (e.g., maximum absolute value)
  - [`ConvPivStrat`](@ref): Convergence-driven strategies (e.g., random sampling)

# Interface

Concrete subtypes should implement:

  - `(strategy::MyPivStrat)(len::Int)`: Create functor for `len` indices
  - `(strategy::MyPivStrat)(idcs::AbstractArray{Int})`: Create functor for index array
"""
abstract type PivStrat end

"""
    PivStratFunctor

Abstract base type for stateful pivoting functors.

Functors maintain state during the pivot selection process (e.g., tracking which
indices have been used). Created by calling a [`PivStrat`](@ref) instance with
index information.

# Interface

Concrete subtypes should implement:

  - `(functor::MyPivStratFunctor)()`: Select initial pivot (no data available)
  - `(functor::MyPivStratFunctor)(rc::AbstractArray)`: Select next pivot based on
    row/column data
"""
abstract type PivStratFunctor end

"""
    GeoPivStrat <: PivStrat

Abstract type for geometric/spatial pivoting strategies.

These strategies select pivots based on spatial/geometric properties rather than
matrix values. Useful when geometric information about rows/columns is available.

# Concrete Types

  - [`FillDistance`](@ref): Maximizes minimum distance to already selected points
  - [`LejaPoints`](@ref): Maximizes product of distances to selected points
"""
abstract type GeoPivStrat <: PivStrat end

"""
    ValuePivStrat <: PivStrat

Abstract type for value-based pivoting strategies.

These strategies select pivots based on matrix element values sampled during the
ACA algorithm. Most common approach for general matrices.

# Concrete Types

  - [`MaximumValue`](@ref): Selects index with maximum absolute value (standard ACA)
  - [`RandomSampling`](@ref): Random selection (for statistical approaches)
"""
abstract type ValuePivStrat <: PivStrat end

"""
    ConvPivStrat <: PivStrat

Abstract type for convergence-driven pivoting strategies.

These strategies adapt their behavior based on convergence information or use
randomization to improve robustness.

# Concrete Types

  - [`RandomSampling`](@ref): Random pivot selection for convergence estimation
"""
abstract type ConvPivStrat <: PivStrat end

"""
    GeoPivStratFunctor <: PivStratFunctor

Abstract type for stateful geometric pivoting functors.
"""
abstract type GeoPivStratFunctor <: PivStratFunctor end

"""
    ConvPivStratFunctor <: PivStratFunctor

Abstract type for stateful convergence-driven pivoting functors.
"""
abstract type ConvPivStratFunctor <: PivStratFunctor end

"""
    ValuePivStratFunctor <: PivStratFunctor

Abstract type for stateful value-based pivoting functors.
"""
abstract type ValuePivStratFunctor <: PivStratFunctor end
