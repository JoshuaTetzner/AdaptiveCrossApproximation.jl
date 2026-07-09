using ProgressMeter: Progress, update!

# Single GPU near-interaction/coupling-matrix entry point: fills every
# preallocated `blocks[i]` in one batched pass via BEAST's
# `gpu_batched_blockassemble!` (integrate + project directly into dense
# per-block output, no global sparse matrix, no host round trip beyond the
# final copy). Reached both via `assemblenears` (through the generic
# `assemble_blocks_sparse(matrix,...) = assemble_blocks(matrix,...)` default
# in `abstractkernelmatrix.jl`, since no GPU-specific `assemble_blocks_sparse`
# method exists anymore) and directly by NestedCrossApproximation's far-field
# batched-bottom-up/coupling-matrix code, which calls `assemble_blocks` on a
# GPU-capable `farmatrix` without going through `assemblenears` at all.
function AdaptiveCrossApproximation.assemble_blocks(
    matrix::AdaptiveCrossApproximation.GPUBEASTKernelMatrix{T},
    blocks::AbstractVector{<:AbstractMatrix{T}},
    rowidcs::AbstractVector{<:AbstractVector{Int}},
    colidcs::AbstractVector{<:AbstractVector{Int}};
    scheduler=nothing,
    pbar=Progress(0; enabled=false),
    kwargs...,
) where {T}
    isempty(blocks) && return blocks
    timing = get(ENV, "ACA_GPU_TIMING", "0") == "1"

    bext = Base.get_extension(BEAST, :BEASTCUDAExt)
    t_assemble = @elapsed bext.gpu_batched_blockassemble!(blocks, matrix.nearassembler, rowidcs, colidcs)
    timing && println("[assemble_blocks] gpu_batched_blockassemble! total: ",
        "$(round(t_assemble; digits=4))s")

    update!(pbar, length(blocks))
    return blocks
end

# No GPUBEASTKernelMatrix-specific `assemblenears` override for
# PreserveSpaceOrder: the generic `AbstractKernelMatrix` method (in
# AdaptiveCrossApproximation's nearinteractions.jl) already delegates block
# filling to `assemble_blocks_sparse`, which generically forwards to
# `assemble_blocks` above.
function AdaptiveCrossApproximation.assemblenears(
    ::AdaptiveCrossApproximation.GPUBEASTKernelMatrix,
    tree,
    ::AdaptiveCrossApproximation.PermuteSpaceInPlace;
    kwargs...,
)
    throw(ArgumentError("GPU near assembly currently supports PreserveSpaceOrder only"))
end
