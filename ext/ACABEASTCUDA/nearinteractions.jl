using ProgressMeter: Progress, update!

# Single GPU near-interaction/coupling-matrix entry point: fills every
# preallocated `blocks[i]` in one batched pass via BEAST's
# `gpu_batched_blockassemble!` (integrate + project directly into dense
# per-block output, no global sparse matrix). The target `blocks` may live on
# host (near/coupling `Vector{Matrix}`) or on device (compression level
# buffer views) - BEAST's device-aware scatter handles both. Reached by the
# HMatrix near-field assembly and directly by NestedCrossApproximation's near
# field and far-field batched-bottom-up/coupling-matrix code, which call
# `assemble_blocks` on a GPU-capable kernel matrix.
function AdaptiveCrossApproximation.assemble_blocks(
    matrix::AdaptiveCrossApproximation.GPUBEASTKernelMatrix{T},
    blocks::AbstractVector{<:AbstractMatrix{T}},
    rowidcs::AbstractVector{<:AbstractVector{Int}},
    colidcs::AbstractVector{<:AbstractVector{Int}};
    pbar=Progress(0; enabled=false),
    kwargs...,
) where {T}
    isempty(blocks) && return blocks
    timing = get(ENV, "ACA_GPU_TIMING", "0") == "1"

    bext = _beastcudaext()
    t_assemble = @elapsed bext.gpu_batched_blockassemble!(
        blocks, matrix.functors, rowidcs, colidcs
    )
    timing && println(
        "[assemble_blocks] gpu_batched_blockassemble! total: ",
        "$(round(t_assemble; digits=4))s",
    )

    update!(pbar, length(blocks))
    return blocks
end
