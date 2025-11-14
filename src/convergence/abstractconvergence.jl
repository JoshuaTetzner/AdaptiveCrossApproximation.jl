"""
    ConvCrit

Abstract base type for convergence criteria.
Convergence criteria determine when to stop the ACA iteration based on approximation quality.
"""
abstract type ConvCrit end

"""
    ConvCritFunctor

Abstract base type for stateful convergence criterion functors.
Used during compression to track convergence state across iterations.
"""
abstract type ConvCritFunctor end

"""
    normF!(convcrit::ConvCritFunctor, rowbuffer, colbuffer, npivot, maxrows, maxcolumns)

Update Frobenius norm estimate for standard ACA.
Incrementally computes squared norm of UV factorization using current pivot and all previous pivots.

# Arguments

  - `convcrit::ConvCritFunctor`: Convergence criterion functor to update
  - `rowbuffer::AbstractMatrix{K}`: Row factor buffer
  - `colbuffer::AbstractMatrix{K}`: Column factor buffer
  - `npivot::Int`: Current pivot index
  - `maxrows::Int`: Number of active rows
  - `maxcolumns::Int`: Number of active columns
"""
function normF!(
    convcrit::ConvCritFunctor,
    rowbuffer::AbstractMatrix{K},
    colbuffer::AbstractMatrix{K},
    npivot::Int,
    maxrows::Int,
    maxcolumns::Int,
) where {K}
    @views convcrit.normUV² +=
        (norm(rowbuffer[npivot, 1:maxcolumns]) * norm(colbuffer[1:maxrows, npivot]))^2

    for j in 1:(npivot - 1)
        @views convcrit.normUV² +=
            2 * real.(
                dot(colbuffer[1:maxrows, npivot], colbuffer[1:maxrows, j]) *
                dot(rowbuffer[npivot, 1:maxcolumns], rowbuffer[j, 1:maxcolumns]),
            )
    end
end

"""
    normF!(convcrit::ConvCritFunctor, rcbuffer::AbstractVector{K}, npivot::Int)

Update running norm estimate for incomplete ACA (iACA).
Computes moving average of row/column norms across pivots.

# Arguments

  - `convcrit::ConvCritFunctor`: Convergence criterion functor to update
  - `rcbuffer::AbstractVector{K}`: Current row or column buffer
  - `npivot::Int`: Current pivot index
"""
function normF!(
    convcrit::ConvCritFunctor, rcbuffer::AbstractVector{K}, npivot::Int
) where {K}
    return convcrit.normUV = ((npivot - 1) * convcrit.normUV + norm(rcbuffer)) / npivot
end
