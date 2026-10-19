"""
    assemblenears(operator, testspace, trialspace, tree, partition; matrixdata, ntasks,
        verbose=true)

Assemble the dense near-field blocks of the terminal blocks `partition` (see
[`interactions`](@ref)) through [`assemble_blocks`](@ref) on `ntasks` tasks.

# Returns

A `Vector{Matrix{T}}` in the order of `partition.nearnodes`.
"""
function assemblenears(
    operator,
    testspace,
    trialspace,
    tree,
    partition;
    matrixdata=defaultmatrixdata(operator, testspace, trialspace),
    ntasks::Integer=Threads.nthreads(),
    verbose::Bool=true,
)
    kernel = AbstractKernelMatrix(operator, testspace, trialspace; matrixdata=matrixdata)
    T = eltype(kernel)
    rows = [values(testtree(tree), τ) for τ in _rownodes(partition.nearptr)]
    cols = [values(trialtree(tree), σ) for σ in partition.nearnodes]
    near = Matrix{T}[zeros(T, length(r), length(c)) for (r, c) in zip(rows, cols)]
    progress = Progress(
        length(near); desc="Assembling near interactions: ", enabled=verbose
    )
    assemble_blocks(kernel, near, rows, cols; ntasks=ntasks, pbar=progress)
    finish!(progress)
    return near
end

"""
    assemblenears(operator, space, tree, partition; matrixdata, ntasks, verbose=true)

Assemble the near-field blocks of the transpose-symmetric terminal blocks
`partition` (see [`symmetricinteractions`](@ref)) on one space. Self blocks are
assembled in full and then made symmetric by copying their upper triangles into
their lower triangles.
"""
function assemblenears(
    operator,
    space,
    tree,
    partition;
    matrixdata=defaultmatrixdata(operator, space, space),
    ntasks::Integer=Threads.nthreads(),
    verbose::Bool=true,
)
    near = assemblenears(
        operator,
        space,
        space,
        tree,
        partition;
        matrixdata=matrixdata,
        ntasks=ntasks,
        verbose=verbose,
    )
    rownodes = _rownodes(partition.nearptr)
    for k in eachindex(near)
        rownodes[k] == partition.nearnodes[k] || continue
        block = near[k]
        for j in axes(block, 2), i in (j + 1):size(block, 1)
            block[i, j] = block[j, i]
        end
    end
    return near
end
