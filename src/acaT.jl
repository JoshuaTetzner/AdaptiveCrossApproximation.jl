"""
    ACAᵀ{RowPivType,ColPivType,ConvCritType}

Column-first variant of adaptive cross approximation.
Starts by selecting columns first, then rows. Dual of standard ACA.

# Fields

  - `rowpivoting::RowPivType`: Strategy for selecting row pivots
  - `columnpivoting::ColPivType`: Strategy for selecting column pivots
  - `convergence::ConvCritType`: Convergence criterion
"""
struct ACAᵀ{RowPivType,ColPivType,ConvCritType}
    rowpivoting::RowPivType
    columnpivoting::ColPivType
    convergence::ConvCritType

    function ACAᵀ(rowpivoting, columnpivoting, convergence)
        return new{typeof(rowpivoting),typeof(columnpivoting),typeof(convergence)}(
            rowpivoting, columnpivoting, convergence
        )
    end
end

function ACAᵀ(;
    tol=1e-4,
    rowpivoting=MaximumValue(),
    columnpivoting=MaximumValue(),
    convergence=FNormEstimator(tol),
)
    return ACAᵀ(rowpivoting, columnpivoting, convergence)
end

function (aca::ACAᵀ{RP,CP,C})(
    rowidcs::AbstractVector{Int}, colidcs::AbstractVector{Int}
) where {RP<:MaximumValue,CP<:MaximumValue,C<:FNormEstimator}
    return ACAᵀ(aca.rowpivoting(rowidcs), aca.columnpivoting(colidcs), aca.convergence())
end

function (aca::ACAᵀ{RP,CP,C})(
    A, rowidcs::AbstractVector{Int}, colidcs::AbstractVector{Int}, maxrank::Int
) where {RP<:PivStrat,CP<:PivStrat,C<:ConvCrit}
    convcrit = _buildconvcrit(aca.convergence, A, rowidcs, colidcs, maxrank)
    rowpiv = _buildpivstrat(aca.rowpivoting, convcrit, rowidcs)
    colpiv = _buildpivstrat(aca.columnpivoting, convcrit, colidcs)

    return ACAᵀ(rowpiv, colpiv, convcrit)
end

function (aca::ACAᵀ{RP,CP,C})(
    A, nrowidcs::Int, ncolidcs::Int, maxrank::Int
) where {RP<:PivStrat,CP<:PivStrat,C<:ConvCrit}
    convcrit = _buildconvcrit(aca.convergence, A, nrowidcs, ncolidcs, maxrank)
    rowpiv = _buildpivstrat(aca.rowpivoting, convcrit, nrowidcs)
    colpiv = _buildpivstrat(aca.columnpivoting, convcrit, ncolidcs)

    return ACAᵀ(rowpiv, colpiv, convcrit)
end

function reset!(
    aca::ACAᵀ{RP,CP,C}, rowidcs::AbstractArray{Int}, colidcs::AbstractArray{Int}
) where {RP<:PivStratFunctor,CP<:PivStratFunctor,C<:ConvCritFunctor}
    reset!(aca.rowpivoting, rowidcs)
    reset!(aca.columnpivoting, colidcs)
    reset!(aca.convergence, rowidcs, colidcs)
    return nothing
end

function (aca::ACAᵀ{RP,CP,C})(
    A,
    colbuffer::AbstractArray{K},
    rowbuffer::AbstractArray{K},
    maxrank::Int;
    rows::Vector{Int}=zeros(Int, maxrank),
    cols::Vector{Int}=zeros(Int, maxrank),
    rowidcs::AbstractVector{Int}=Vector(1:size(colbuffer, 1)),
    colidcs::AbstractVector{Int}=Vector(1:size(rowbuffer, 2)),
) where {K,RP<:PivStrat,CP<:PivStrat,C<:ConvCrit}
    return aca(A, rowidcs, colidcs, maxrank)(
        A, colbuffer, rowbuffer, rows, cols, rowidcs, colidcs, maxrank
    )
end

function (aca::ACAᵀ{RP,CP,C})(
    A,
    colbuffer::AbstractArray{K},
    rowbuffer::AbstractArray{K},
    maxrank::Int;
    rows::Vector{Int}=zeros(Int, maxrank),
    cols::Vector{Int}=zeros(Int, maxrank),
    rowidcs::AbstractVector{Int}=Vector(1:size(colbuffer, 1)),
    colidcs::AbstractVector{Int}=Vector(1:size(rowbuffer, 2)),
) where {K,RP<:PivStratFunctor,CP<:PivStratFunctor,C<:ConvCritFunctor}
    reset!(aca, rowidcs, colidcs)
    return aca(A, colbuffer, rowbuffer, rows, cols, rowidcs, colidcs, maxrank)
end

"""
    (aca::ACAᵀ)(A, colbuffer, rowbuffer, rows, cols, rowidcs, colidcs, maxrank)

Compute column-first ACA approximation with preallocated buffers (main computational routine).

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
function (aca::ACAᵀ)(
    A,
    colbuffer::AbstractArray{K},
    rowbuffer::AbstractArray{K},
    rows::AbstractVector{Int},
    cols::AbstractVector{Int},
    rowidcs::AbstractVector{Int},
    colidcs::AbstractVector{Int},
    maxrank::Int,
) where {K}
    maxrows = size(colbuffer, 1)
    maxcols = size(rowbuffer, 2)
    npivot = 1
    nextcol = aca.columnpivoting()
    cols[1] = colidcs[nextcol]
    nextrc!(
        view(colbuffer, 1:maxrows, npivot:npivot),
        A,
        view(rowidcs, 1:maxrows),
        view(colidcs, 1:1),
    )
    @views nextrow = aca.rowpivoting(colbuffer[1:maxrows, npivot])
    rows[npivot] = rowidcs[nextrow]
    if colbuffer[nextrow, npivot] != 0.0
        view(colbuffer, 1:maxrows, npivot) ./= view(colbuffer, nextrow, npivot)
    end
    nextrc!(
        view(rowbuffer, npivot:npivot, 1:maxcols),
        A,
        view(rowidcs, nextrow:nextrow),
        view(colidcs, 1:maxcols),
    )

    # conv is true until convergence is reached
    npivot, conv = aca.convergence(rowbuffer, colbuffer, npivot, maxrows, maxcols)

    while conv && npivot < maxrank
        npivot += 1
        @views nextcol = aca.columnpivoting(rowbuffer[max(1, npivot - 1), 1:maxcols])
        cols[npivot] = colidcs[nextcol]
        nextrc!(
            view(colbuffer, 1:maxrows, npivot:npivot),
            A,
            view(rowidcs, 1:maxrows),
            view(colidcs, nextcol:nextcol),
        )

        for k in 1:(npivot - 1)
            for kk in 1:maxrows
                colbuffer[kk, npivot] -= rowbuffer[k, nextcol] * colbuffer[kk, k]
            end
        end

        @views nextrow = aca.rowpivoting(colbuffer[1:maxrows, npivot])
        rows[npivot] = rowidcs[nextrow]
        if colbuffer[nextrow, npivot] != 0.0
            view(colbuffer, 1:maxrows, npivot) ./= view(colbuffer, nextrow, npivot)
            nextrc!(
                view(rowbuffer, npivot:npivot, 1:maxcols),
                A,
                view(rowidcs, nextrow:nextrow),
                view(colidcs, 1:maxcols),
            )
        end

        for k in 1:(npivot - 1)
            for kk in 1:maxcols
                rowbuffer[npivot, kk] -= rowbuffer[k, kk] * colbuffer[nextrow, k]
            end
        end

        npivot, conv = aca.convergence(rowbuffer, colbuffer, npivot, maxrows, maxcols)
    end

    return npivot
end

"""
    acaᵀ(M; tol=1e-4, rowpivoting=MaximumValue(), columnpivoting=MaximumValue(),
        convergence=FNormEstimator(tol), maxrank=40, svdrecompress=false)

Compute column-first adaptive cross approximation of matrix `M` returning low-rank factors.

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
function acaᵀ(
    M::AbstractMatrix{K};
    tol=1e-4,
    rowpivoting=MaximumValue(),
    columnpivoting=MaximumValue(),
    convergence=FNormEstimator(tol),
    maxrank=40,
    svdrecompress=false,
) where {K}
    compressor = ACAᵀ(rowpivoting, columnpivoting, convergence)
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
