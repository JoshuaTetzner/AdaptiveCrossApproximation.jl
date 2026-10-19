# Example-only adapter: assemble a sum or a small block system directly into one
# HMatrix. Integral terms cache their quadrature data; local terms stay sparse.
module BEMHMatrixExample

using AdaptiveCrossApproximation
using BEAST

const ACA = AdaptiveCrossApproximation
export GalerkinTerm, GalerkinOperator

struct GalerkinTerm{T,N,F}
    near::N
    far::F
end
Base.eltype(::GalerkinTerm{T}) where {T} = T

function GalerkinTerm(op::BEAST.IntegralOperator, testspace, trialspace)
    nearquad = BEAST.defaultquadstrat(op, testspace, trialspace)
    farquad = ACA.tofarquadstrat(nearquad)
    near = BEAST.blockassembler(op, testspace, trialspace; quadstrat=nearquad)
    far = if farquad == nearquad
        near
    else
        BEAST.blockassembler(op, testspace, trialspace; quadstrat=farquad)
    end
    return GalerkinTerm{BEAST.scalartype(op),typeof(near),typeof(far)}(near, far)
end

function GalerkinTerm(op::BEAST.LocalOperator, testspace, trialspace)
    matrix = BEAST.assemble(op, testspace, trialspace)
    return GalerkinTerm{eltype(matrix),typeof(matrix),typeof(matrix)}(matrix, matrix)
end

struct GalerkinKernel{T,B} <: ACA.AbstractKernelMatrix{T}
    blocks::B
    blocksize::Int
end
Base.size(K::GalerkinKernel) =
    (length(K.blocks) * K.blocksize, length(K.blocks) * K.blocksize)
Base.size(K::GalerkinKernel, dim::Integer) = size(K)[dim]

struct GalerkinOperator{N,F}
    near::N
    far::F
end

# blocks is a tuple of block rows. Each block holds (coefficient, GalerkinTerm)
# pairs; every component has blocksize unknowns, in the original basis order.
function GalerkinOperator(blocks::Tuple, blocksize::Int)
    @assert all(length(row) == length(blocks) for row in blocks)
    T = promote_type(
        [
            promote_type(typeof(alpha), eltype(term)) for row in blocks for terms in row for
            (alpha, term) in terms
        ]...,
    )
    near = map(row -> map(terms -> map(t -> (t[1], t[2].near), terms), row), blocks)
    far = map(row -> map(terms -> map(t -> (t[1], t[2].far), terms), row), blocks)
    return GalerkinOperator(
        GalerkinKernel{T,typeof(near)}(near, blocksize),
        GalerkinKernel{T,typeof(far)}(far, blocksize),
    )
end

function ACA.AbstractKernelMatrix(
    op::GalerkinOperator, testspace, trialspace; matrixdata=nothing
)
    return matrixdata === :far ? op.far : op.near
end

# The double-layer terms can have zero initial samples. Reuse BEAST's robust
# ACA configuration rather than stopping at the first zero row.
function ACA.defaultcompressor(::GalerkinOperator, testspace, trialspace; tol=1e-4)
    convergence = ACA.CombinedConvCrit([ACA.FNormEstimator(tol), ACA.RandomSampling(; tol)])
    rowpivoting = ACA.CombinedPivStrat([ACA.MaximumValue(), ACA.RandomSamplingPivoting(1)])
    return ACA.ACA(; rowpivoting, convergence)
end

struct ScaledBlockStore{M,T}
    matrix::M
    alpha::T
end
function (store::ScaledBlockStore)(value, row, col)
    @inbounds store.matrix[row, col] += store.alpha * value
    return nothing
end

addterm!(block, assembler, rows, cols, alpha) =
    assembler(rows, cols, ScaledBlockStore(block, alpha))

function addterm!(block, matrix::AbstractMatrix, rows, cols, alpha)
    for (j, col) in enumerate(cols), (i, row) in enumerate(rows)
        @inbounds block[i, j] += alpha * matrix[row, col]
    end
    return nothing
end

function component_indices(ids, component, blocksize)
    offset = (component - 1) * blocksize
    slots = findall(id -> offset < id <= offset + blocksize, ids)
    return slots, [ids[slot] - offset for slot in slots]
end

function (K::GalerkinKernel)(block, rows, cols)
    if length(K.blocks) == 1
        for (alpha, assembler) in K.blocks[1][1]
            addterm!(block, assembler, rows, cols, alpha)
        end
        return nothing
    end
    rowparts = ntuple(i -> component_indices(rows, i, K.blocksize), length(K.blocks))
    colparts = ntuple(i -> component_indices(cols, i, K.blocksize), length(K.blocks))
    for (i, blockrow) in enumerate(K.blocks), (j, terms) in enumerate(blockrow)
        isempty(rowparts[i][1]) && continue
        isempty(colparts[j][1]) && continue
        target = view(block, rowparts[i][1], colparts[j][1])
        for (alpha, assembler) in terms
            addterm!(target, assembler, rowparts[i][2], colparts[j][2], alpha)
        end
    end
    return nothing
end

indices(i::Integer) = i:i
indices(ids) = ids
ACA.nextrc!(block, K::GalerkinKernel, rows, cols) = K(block, indices(rows), indices(cols))

end
