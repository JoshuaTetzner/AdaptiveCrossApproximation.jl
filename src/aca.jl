"""
    ACA{RowPivType,ColPivType,ConvCritType}

Adaptive Cross Approximation (ACA) compressor for low-rank matrix approximation.

Computes `M ≈ U * V` by iteratively selecting rows and columns (pivots) until a
convergence criterion is met. The algorithm starts with row, samples it to select
a column pivot, then alternates between row and column selection.

# Fields

  - `rowpivoting::RowPivType`: Strategy for selecting row pivots
  - `columnpivoting::ColPivType`: Strategy for selecting column pivots
  - `convergence::ConvCritType`: Convergence criterion to stop iterations
"""
struct ACA{RowPivType,ColPivType,ConvCritType}
    rowpivoting::RowPivType
    columnpivoting::ColPivType
    convergence::ConvCritType

    function ACA(rowpivoting, columnpivoting, convergence)
        return new{typeof(rowpivoting),typeof(columnpivoting),typeof(convergence)}(
            rowpivoting, columnpivoting, convergence
        )
    end
end

function ACA(;
    tol=1e-4,
    rowpivoting=MaximumValue(),
    columnpivoting=MaximumValue(),
    convergence=FNormEstimator(tol),
)
    return ACA(rowpivoting, columnpivoting, convergence)
end

function (aca::ACA{RP,CP,C})(
    rowidcs::AbstractArray{Int}, colidcs::AbstractArray{Int}
) where {RP<:MaximumValue,CP<:MaximumValue,C<:FNormEstimator}
    return ACA(aca.rowpivoting(rowidcs), aca.columnpivoting(colidcs), aca.convergence())
end

function (aca::ACA{RP,CP,C})(
    A::Union{AbstractMatrix,AbstractKernelMatrix},
    rowidcs::AbstractArray{Int},
    colidcs::AbstractArray{Int},
    maxrank::Int,
) where {RP<:PivStrat,CP<:PivStrat,C<:ConvCrit}
    rowpiv = _buildpivstrat(aca.rowpivoting, aca, rowidcs)
    colpiv = _buildpivstrat(aca.columnpivoting, aca, colidcs)
    convcrit = _buildconvcrit(aca.convergence, A, rowidcs, colidcs, maxrank)

    return ACA(rowpiv, colpiv, convcrit)
end

function reset!(
    aca::ACA{RP,CP,C}, rowidcs::AbstractArray{Int}, colidcs::AbstractArray{Int}
) where {RP<:PivStratFunctor,CP<:PivStratFunctor,C<:ConvCritFunctor}
    reset!(aca.rowpivoting, rowidcs)
    reset!(aca.columnpivoting, colidcs)
    reset!(aca.convergence, rowidcs, colidcs)
    return nothing
end

#=function (aca::ACA{RP,CP,C})(
    A,
    colbuffer::AbstractArray{K},
    rowbuffer::AbstractArray{K},
    maxrank::Int;
    rows=zeros(Int, maxrank),
    cols=zeros(Int, maxrank),
    rowidcs=Vector(1:size(colbuffer, 1)),
    colidcs=Vector(1:size(rowbuffer, 2)),
) where {K,RP<:PivStrat,CP<:PivStrat,C<:ConvCrit}
    return aca(A, rowidcs, colidcs, maxrank)(
        A, colbuffer, rowbuffer, rows, cols, rowidcs, colidcs, maxrank
    )
end=#

function (aca::ACA{RP,CP,C})(
    A,
    colbuffer::AbstractArray{K},
    rowbuffer::AbstractArray{K},
    maxrank::Int;
    rows=zeros(Int, maxrank),
    cols=zeros(Int, maxrank),
    rowidcs=Vector(1:size(colbuffer, 1)),
    colidcs=Vector(1:size(rowbuffer, 2)),
) where {K,RP<:PivStratFunctor,CP<:PivStratFunctor,C<:ConvCritFunctor}
    reset!(aca, rowidcs, colidcs)
    return aca(A, colbuffer, rowbuffer, rows, cols, rowidcs, colidcs, maxrank)
end

"""
    (aca::ACA)(A, colbuffer, rowbuffer, rows, cols, rowidcs, colidcs, maxrank)

Compute ACA approximation with preallocated buffers (main computational routine).

Fills `colbuffer` and `rowbuffer` with low-rank factors U and V such that
`A[rowidcs, colidcs] ≈ U * V`. Uses deflation to ensure orthogonality of pivots.

# Arguments

  - `A`: Matrix or matrix-like object (must support `nextrc!` interface)
  - `colbuffer::AbstractArray{K}`: Buffer for U factors, size `(length(rowidcs), maxrank)`
  - `rowbuffer::AbstractArray{K}`: Buffer for V factors, size `(maxrank, length(colidcs))`
  - `rows::Vector{Int}`: Storage for selected row indices
  - `cols::Vector{Int}`: Storage for selected column indices
  - `rowidcs::Vector{Int}`: Global row indices of the block to compress
  - `colidcs::Vector{Int}`: Global column indices of the block to compress
  - `maxrank::Int`: Maximum number of pivots (hard limit on rank)

# Returns

  - `npivot::Int`: Number of pivots computed (≤ maxrank). The approximation is
    `A[rowidcs, colidcs] ≈ colbuffer[:, 1:npivot] * rowbuffer[1:npivot, :]`
"""
function (aca::ACA)(
    A,
    colbuffer::AbstractArray{K},
    rowbuffer::AbstractArray{K},
    rows::AbstractVector{Int},
    cols::AbstractVector{Int},
    rowidcs::AbstractVector{Int},
    colidcs::AbstractVector{Int},
    maxrank::Int,
) where {K}
    maxrows = length(rowidcs)
    maxcols = length(colidcs)
    npivot = 1
    nextrow = aca.rowpivoting()
    rows[1] = rowidcs[nextrow]
    nextrc!(
        view(rowbuffer, npivot:npivot, 1:maxcols),
        A,
        view(rowidcs, 1:1),
        view(colidcs, 1:maxcols),
    )

    @views nextcolumn = aca.columnpivoting(rowbuffer[npivot, 1:maxcols])

    cols[npivot] = colidcs[nextcolumn]

    if rowbuffer[npivot, nextcolumn] != 0.0
        view(rowbuffer, npivot, 1:maxcols) ./= view(rowbuffer, npivot, nextcolumn)
    end
    nextrc!(
        view(colbuffer, 1:maxrows, npivot:npivot),
        A,
        view(rowidcs, 1:maxrows),
        view(colidcs, nextcolumn:nextcolumn),
    )

    # conv is true until convergence is reached
    npivot, conv = aca.convergence(rowbuffer, colbuffer, npivot, maxrows, maxcols)

    while conv && npivot < maxrank
        npivot += 1
        @views nextrow = aca.rowpivoting(colbuffer[1:maxrows, max(1, npivot - 1)])
        rows[npivot] = rowidcs[nextrow]
        nextrc!(
            view(rowbuffer, npivot:npivot, 1:maxcols),
            A,
            view(rowidcs, nextrow:nextrow),
            view(colidcs, 1:maxcols),
        )

        for k in 1:(npivot - 1)
            for kk in 1:maxcols
                rowbuffer[npivot, kk] -= colbuffer[nextrow, k] * rowbuffer[k, kk]
            end
        end

        @views nextcolumn = aca.columnpivoting(rowbuffer[npivot, 1:maxcols])
        cols[npivot] = colidcs[nextcolumn]
        if rowbuffer[npivot, nextcolumn] != 0.0
            view(rowbuffer, npivot, 1:maxcols) ./= view(rowbuffer, npivot, nextcolumn)
            nextrc!(
                view(colbuffer, 1:maxrows, npivot:npivot),
                A,
                view(rowidcs, 1:maxrows),
                view(colidcs, nextcolumn:nextcolumn),
            )
        end

        for k in 1:(npivot - 1)
            for kk in 1:maxrows
                colbuffer[kk, npivot] -= colbuffer[kk, k] * rowbuffer[k, nextcolumn]
            end
        end
        npivot, conv = aca.convergence(rowbuffer, colbuffer, npivot, maxrows, maxcols)
    end
    return npivot
end

"""
    aca(M; tol=1e-4, rowpivoting=MaximumValue(), columnpivoting=MaximumValue(),
        convergence=FNormEstimator(tol), maxrank=40, svdrecompress=false)

Compute adaptive cross approximation of matrix `M` returning low-rank factors.

High-level convenience function that automatically allocates buffers and returns
`U, V` such that `M ≈ U * V`.

# Arguments

  - `M::AbstractMatrix{K}`: Matrix to approximate

# Keyword Arguments

  - `tol::Real = 1e-4`: Approximation tolerance
  - `rowpivoting = MaximumValue()`: Row pivot selection strategy
  - `columnpivoting = MaximumValue()`: Column pivot selection strategy
  - `convergence = FNormEstimator(tol)`: Convergence criterion
  - `maxrank::Int = 40`: Maximum rank (hard limit)
  - `svdrecompress::Bool = false`: Apply SVD-based recompression to reduce rank further

# Returns

  - `U::Matrix{K}`: Left factor, size `(size(M,1), r)` where `r ≤ maxrank`
  - `V::Matrix{K}`: Right factor, size `(r, size(M,2))`

Satisfies `M ≈ U * V` with `norm(M - U*V) / norm(M) ≲ tol` (if maxrank sufficient).

# SVD Recompression

When `svdrecompress=true`, performs QR-SVD recompression: computes `M ≈ U*V`, then
`U = Q*R`, `R*V = Û*Σ*V̂ᵀ`, truncates small singular values, and returns optimal
rank factors at the cost of additional computation.
"""
function aca(
    M::AbstractMatrix{K};
    tol=1e-4,
    rowpivoting=MaximumValue(),
    columnpivoting=MaximumValue(),
    convergence=FNormEstimator(tol),
    maxrank::Int=40,
    svdrecompress=false,
) where {K}
    compressor = ACA(rowpivoting, columnpivoting, convergence)
    rowbuffer = zeros(K, maxrank, size(M, 2))
    colbuffer = zeros(K, size(M, 1), maxrank)
    npivots = compressor(M, colbuffer, rowbuffer, maxrank)
    if svdrecompress
        @views Q, R = qr(colbuffer[1:size(M, 1), 1:npivots])
        @views U, s, V = svd(R * rowbuffer[1:npivots, 1:size(M, 2)])

        opt_r = length(s)
        for i in eachindex(s)
            if s[i] < tolerance(convergence) * s[1]
                opt_r = i
                break
            end
        end

        A = (Q * U)[1:size(M, 1), 1:opt_r]
        B = (diagm(s) * V')[1:opt_r, 1:size(M, 2)]

        return A, B
    else
        return colbuffer[1:size(M, 1), 1:npivots], rowbuffer[1:npivots, 1:size(M, 2)]
    end
end
