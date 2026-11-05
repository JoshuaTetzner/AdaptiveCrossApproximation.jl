using OhMyThreads

"""
    assemble_blocks(matrix::AbstractKernelMatrix, blocks, rowidcs, colidcs; ntasks, pbar) -> blocks

Fill each preallocated `blocks[i]` (sized `(length(rowidcs[i]), length(colidcs[i]))`)
with the requested kernel entries.
The built-in kernel backends accumulate into the supplied blocks, so callers should
normally provide zero-initialized storage.

`rowidcs`/`colidcs` need not describe disjoint or contiguous index ranges —
they can be the cluster indices of near-field blocks (see
[`interactions`](@ref)) but also scattered pivot-index blocks (e.g. the
far-field coupling blocks of a nested cross approximation), since both are
just lists of (row-index-list, column-index-list) requests against the same
underlying kernel.

Taking `blocks` as an argument (rather than allocating and returning it) lets
callers own the storage layout (e.g. the near blocks of an [`HMatrix`](@ref)) and
lets backends override just the fill strategy — e.g. a GPU backend can force a
serial loop, or batch everything through one combined pass — without
duplicating the surrounding preallocation/storage-construction code.

Default (any `AbstractKernelMatrix`) implementation: a plain per-block loop on
`ntasks` parallel tasks (default `Threads.nthreads()`).
"""
function assemble_blocks(
    matrix::AbstractKernelMatrix{T},
    blocks::AbstractVector{<:AbstractMatrix{T}},
    rowidcs::AbstractVector{<:AbstractVector{Int}},
    colidcs::AbstractVector{<:AbstractVector{Int}};
    ntasks::Integer=Threads.nthreads(),
    pbar=Progress(0; enabled=false),
) where {T}
    scheduler = _scheduler(ntasks)
    @tasks for i in eachindex(blocks)
        @set scheduler = scheduler
        matrix(blocks[i], rowidcs[i], colidcs[i])
        next!(pbar)
    end
    return blocks
end
