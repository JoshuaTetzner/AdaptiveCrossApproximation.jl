# Concatenate the per-node lists `nested` into `(ptr, nodes)`: the entries of
# node `τ` are `nodes[ptr[τ]:(ptr[τ + 1] - 1)]`.
function linearizestorage(nested::Vector{Vector{Int}})
    ptr = Vector{Int}(undef, length(nested) + 1)
    ptr[1] = 1
    for (τ, entries) in enumerate(nested)
        ptr[τ + 1] = ptr[τ] + length(entries)
    end
    nodes = Vector{Int}(undef, ptr[end] - 1)
    for (τ, entries) in enumerate(nested)
        copyto!(nodes, ptr[τ], entries, 1, length(entries))
    end
    return ptr, nodes
end

function _checkntasks(ntasks)
    ntasks > 0 || throw(ArgumentError("ntasks must be positive"))
    return Int(ntasks)
end

"""
    _scheduler(ntasks)

Scheduler of the parallel loops over independent items (assembly, matrix-vector
products, recompression) on `ntasks` tasks: a `GreedyScheduler`, where each task
takes the next item as soon as it is done, which balances items of uneven cost.
`ntasks = 1` runs serially on the calling task. The public functions only take
`ntasks`; operations with dependencies use a task graph instead.
"""
function _scheduler(ntasks)
    return _checkntasks(ntasks) == 1 ? SerialScheduler() : GreedyScheduler(; ntasks)
end

"""
    _nchunks(ntasks)

Number of cost-balanced chunks of a loop over many cheap items on `ntasks`
tasks: eight per task, so that the greedy scheduler can balance the remaining
cost differences, and one for a single task.
"""
_nchunks(ntasks) = _checkntasks(ntasks) == 1 ? 1 : 8 * ntasks
