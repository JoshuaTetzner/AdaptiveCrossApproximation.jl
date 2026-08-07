"""
    FullPivoting <: ValuePivStrat

Pivoting strategy that selects the entry of maximum absolute value across the entire
remaining (unfactored) submatrix.

Unlike [`MaximumValue`](@ref), which only searches a single sampled row or column,
full pivoting requires the whole matrix to be available up front: see the
`ACA{FullPivoting,FullPivoting,FNormEstimator}` callable in `fullpivotedaca.jl`, which
has no `nextrc!`-based partial-sampling path, and its batched GPU counterpart in the
`ACACUDA` extension.
"""
struct FullPivoting <: ValuePivStrat end
