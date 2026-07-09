function (aca::ACA{RP,CP,C})(
    A::AbstractMatrix{T}, maxrank::Int=min(size(A)...)
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
        norm² <= tol² * norm₀² && break
    end

    return rank, rowpivots[1:rank], colpivots[1:rank]
end
