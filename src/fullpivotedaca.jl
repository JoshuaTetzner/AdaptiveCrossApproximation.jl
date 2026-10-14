"""
    (aca::ACA{FullPivoting,FullPivoting,FNormEstimator})(A, maxrank=min(size(A)...); sampled=false)

Compute a full-pivoting ACA factorization of the dense matrix `A`, in place.

At each step, selects the entry of largest absolute value in the remaining
(unfactored) submatrix as the next pivot, swaps it onto the diagonal, and eliminates
it with a rank-1 Schur update — the direct, whole-matrix counterpart of
`MaximumValue`-based ACA. Used wherever the whole matrix is available up front, e.g.
as the CPU reference for the batched GPU kernel in the `ACACUDA` extension.

With `sampled=true`, stops when
`norm(residual)/norm(original) * sqrt(s/(s-rank)) <= tol`, where `s` is the
sampled column count. This estimates the full block's relative Frobenius error
assuming representative remaining residual entries and a far field much larger
than the sample. Pivot rows cancel because all rows are sampled.
The default `sampled=false` uses the ordinary relative Frobenius criterion.

# Arguments

  - `A::AbstractMatrix{T}`: matrix to factor; overwritten in place with the working
    Schur-complement state.
  - `maxrank::Int`: hard cap on the number of pivots.
  - `sampled::Bool`: rescale the residual of representative column samples.

# Returns

  - `rank::Int`: number of pivots selected (`≤ maxrank`).
  - `rowpivots::Vector{Int}`, `colpivots::Vector{Int}`: selected row/column indices,
    in selection order.
"""
function (aca::ACA{RP,CP,C})(
    A::AbstractMatrix{T}, maxrank::Int=min(size(A)...); sampled::Bool=false
) where {T,RP<:FullPivoting,CP<:FullPivoting,C<:FNormEstimator}
    m, n = size(A)
    rowpivots = collect(1:m)
    colpivots = collect(1:n)
    norm₀² = sum(abs2, A)
    tol² = aca.convergence.tol^2
    rank = 0

    for k in 1:min(maxrank, m, n)
        pivotvalue² = zero(real(T))
        pivotrow = k
        pivotcol = k
        @inbounds for j in k:n, i in k:m
            value² = abs2(A[i, j])
            if value² > pivotvalue²
                pivotvalue² = value²
                pivotrow = i
                pivotcol = j
            end
        end
        iszero(pivotvalue²) && break

        @inbounds if pivotrow != k
            for j in 1:n
                A[k, j], A[pivotrow, j] = A[pivotrow, j], A[k, j]
            end
            rowpivots[k], rowpivots[pivotrow] = rowpivots[pivotrow], rowpivots[k]
        end
        @inbounds if pivotcol != k
            for i in 1:m
                A[i, k], A[i, pivotcol] = A[i, pivotcol], A[i, k]
            end
            colpivots[k], colpivots[pivotcol] = colpivots[pivotcol], colpivots[k]
        end

        pivotinverse = inv(A[k, k])
        norm² = zero(real(T))
        @inbounds for j in (k + 1):n
            scale = A[k, j] * pivotinverse
            @simd for i in (k + 1):m
                A[i, j] -= A[i, k] * scale
                norm² += abs2(A[i, j])
            end
        end

        rank = k
        k == min(m, n) && break  # No residual entries remain; avoid 0/0.
        correction² = sampled ? real(T)(n) / real(T)(n - k) : one(real(T))
        norm² * correction² <= tol² * norm₀² && break
    end

    return rank, rowpivots[1:rank], colpivots[1:rank]
end
