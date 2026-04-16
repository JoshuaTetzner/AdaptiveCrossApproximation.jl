
"""
    farinteractions(tree; args...)

Extract far-field (inadmissible) block pairs requiring low-rank compression.

Should be implemented by tree backends (e.g., H2Trees) to return index ranges
for pairs of clusters that do NOT satisfy the admissibility criterion. These
blocks are compressed via ACA or similar algorithms.

# Arguments

  - `tree`: Hierarchical tree object implementing the tree interface
  - `args...`: Additional keyword arguments passed by the assembly routine (e.g., `isnear`)

# Returns

  - `(values, farptr, farvalues)`: Triple of row indices, far-field pointers, and column ranges

# Notes

Implemented by tree backend extensions. The pointer array `farptr` enables
efficient lookup of far-field couples for each node, reducing access overhead
during level-by-level assembly.
"""
function farinteractions(tree; args...)
    return error("Needs to be implemented for $(typeof(tree))")
end

"""
    farinteractions_consecutive(tree; args...)

Extract far-field blocks with consecutive storage layout.

Variant used when spaces are permuted to align with tree ordering. Returns
far-field block structure compatible with permuted space layout.

# Arguments

  - `tree`: Hierarchical tree object
  - `args...`: Additional keyword arguments (e.g., `isnear`)

# Returns

  - `(values, farptr, farvalues)`: Index structure for consecutive far-field layout

# Notes

Implemented by tree backend extensions. Must account for space permutation
applied by `PermuteSpaceInPlace()` ordering strategy.
"""
function farinteractions_consecutive(tree; args...)
    return error("Needs to be implemented for $(typeof(tree))")
end

"""
    assemblefars(operator, testspace, trialspace, tree, ::PreserveSpaceOrder; kwargs...)

Assemble far-field blocks without reordering test and trial spaces.

Compresses inadmissible cluster pairs using ACA-style algorithms, maintaining the
original space ordering. Produces a collection of level-by-level block-sparse matrices.

# Arguments

  - `operator`: Operator/kernel for matrix entry evaluation
  - `testspace`: Test space (row basis/evaluation points)
  - `trialspace`: Trial space (column basis/evaluation points)
  - `tree`: Hierarchical tree structure controlling block layout
  - `::PreserveSpaceOrder`: Space ordering strategy marker
  - `compressor = ACA()`: Low-rank compressor algorithm
  - `isnear`: Admissibility predicate (default: `isnear()`)
  - `matrixdata`: Assembly data passed to kernel matrix (optional)
  - `maxrank = 50`: Maximum rank for compressed blocks
  - `scheduler`: Thread scheduler (default: `SerialScheduler()`)

# Returns

  - `Vector{BlockSparseMatrix}`: Collection of level-wise compressed blocks

# Notes

Called by `HMatrix` constructor when `spaceordering=PreserveSpaceOrder()`. Iterates
over tree levels, compressing each level's far-field interactions independently.
Returns one sparse matrix per level, indexed by tree depth.
"""
function assemblefars(
    operator,
    testspace,
    trialspace,
    tree,
    ::PreserveSpaceOrder;
    compressor=ACA(),
    isnear=isnear(),
    matrixdata=defaultfarmatrixdata(operator, testspace, trialspace),
    maxrank=50,
    scheduler=SerialScheduler(),
)
    kernelmatrix = AbstractKernelMatrix(
        operator, testspace, trialspace; matrixdata=matrixdata
    )
    values, farptr, farvalues = farinteractions(tree; isnear=isnear)

    (farptr[end] == 1) && return [
        BlockSparseMatrix(
            Matrix{eltype(kernelmatrix)}[],
            Vector{Int}[],
            Vector{Int}[],
            size(kernelmatrix),
        ),
    ]

    blocks = Vector{LowRankMatrix{eltype(kernelmatrix)}}(undef, length(farvalues))
    colbuffer = zeros(eltype(kernelmatrix), length(testspace), maxrank)
    farinteractionmatrix = BlockSparseMatrix[]
    for level in levels(testtree(tree))
        levelnodes = collect(LevelIterator(testtree(tree), level))
        rbsize, cbsize = buffersize(values, farptr, farvalues, levelnodes)
        cbsize == 0 && continue
        @tasks for node in levelnodes
            @set scheduler = scheduler
            @local begin
                rowbuffer = zeros(eltype(kernelmatrix), maxrank, cbsize)
                localcompressor = compressor(kernelmatrix, rbsize, cbsize, maxrank)
            end
            for faridx in farptr[node]:(farptr[node + 1] - 1)
                npivots = localcompressor(
                    kernelmatrix,
                    view(colbuffer, values[node], 1:maxrank),
                    rowbuffer,
                    maxrank;
                    rowidcs=values[node],
                    colidcs=farvalues[faridx],
                )
                npivots == maxrank && @warn "Maximum rank block"
                blocks[faridx] = LowRankMatrix(
                    colbuffer[values[node], 1:npivots],
                    rowbuffer[1:npivots, 1:length(farvalues[faridx])],
                )
                colbuffer[values[node], 1:npivots] .= eltype(kernelmatrix)(0)
                rowbuffer[1:npivots, 1:length(farvalues[faridx])] .= eltype(kernelmatrix)(0)
            end
        end
        levelvals, levelfarvals, levelidcs = blockvalues(
            values, farptr, farvalues, levelnodes
        )
        push!(
            farinteractionmatrix,
            BlockSparseMatrix(
                view(blocks, levelidcs),
                levelvals,
                levelfarvals,
                size(kernelmatrix);
                scheduler=scheduler,
            ),
        )
    end

    return farinteractionmatrix
end

"""
    assemblefars(operator, testspace, trialspace, tree, ::PermuteSpaceInPlace; kwargs...)

Assemble far-field blocks with tree-aligned space reordering.

Compresses inadmissible cluster pairs using ACA-style algorithms, with both spaces
permuted to align with the hierarchical tree structure. Produces a specialized storage
format optimized for the reordered layout.

# Arguments

  - `operator`: Operator/kernel for matrix entry evaluation
  - `testspace`: Test space (row basis) — will be reordered
  - `trialspace`: Trial space (column basis) — will be reordered
  - `tree`: Hierarchical tree structure
  - `::PermuteSpaceInPlace`: Space ordering strategy marker
  - `compressor = ACA()`: Low-rank compressor algorithm
  - `isnear`: Admissibility predicate (default: `isnear()`)
  - `matrixdata`: Assembly data passed to kernel matrix (optional)
  - `maxrank = 50`: Maximum rank for compressed blocks
  - `scheduler`: Thread scheduler (default: `SerialScheduler()`)

# Returns

  - `Vector{VariableBlockCompressedRowStorage}`: Level-wise compressed blocks in permuted layout

# Notes

Called by `HMatrix` constructor when `spaceordering=PermuteSpaceInPlace()`. The spaces
are reordered to match tree leaf ordering before compression, improving cache locality
and reducing fragmentation. Returns specialized block storage optimized for permuted layout.
"""
function assemblefars(
    operator,
    testspace,
    trialspace,
    tree,
    ::PermuteSpaceInPlace;
    compressor=ACA(),
    isnear=isnear(),
    matrixdata=defaultfarmatrixdata(operator, testspace, trialspace),
    maxrank=50,
    scheduler=SerialScheduler(),
)
    kernelmatrix = AbstractKernelMatrix(
        operator, testspace, trialspace; matrixdata=matrixdata
    )
    values, farptr, farvalues = farinteractions_consecutive(tree; isnear=isnear)

    blocks = Vector{LowRankMatrix{eltype(kernelmatrix)}}(undef, length(farvalues))
    colbuffer = zeros(eltype(kernelmatrix), length(testspace), maxrank)
    farinteractionmatrix = VariableBlockCompressedRowStorage[]
    for level in levels(testtree(tree))
        levelnodes = collect(LevelIterator(testtree(tree), level))
        rbsize, cbsize = buffersize(values, farptr, farvalues, levelnodes)
        @tasks for node in levelnodes
            @set scheduler = scheduler
            @local begin
                rowbuffer = zeros(eltype(kernelmatrix), maxrank, cbsize)
                localcompressor = compressor(kernelmatrix, rbsize, cbsize, maxrank)
            end
            for faridx in farptr[node]:(farptr[node + 1] - 1)
                npivots = localcompressor(
                    kernelmatrix,
                    view(colbuffer, values[node], 1:maxrank),
                    rowbuffer,
                    maxrank;
                    rowidcs=values[node],
                    colidcs=farvalues[faridx],
                )
                blocks[faridx] = LowRankMatrix(
                    colbuffer[values[node], 1:npivots],
                    rowbuffer[1:npivots, 1:length(farvalues[faridx])],
                )
                colbuffer[values[node], 1:npivots] .= eltype(kernelmatrix)(0)
                rowbuffer[1:npivots, 1:length(farvalues[faridx])] .= eltype(kernelmatrix)(0)
            end
        end

        faridcs = [i for idx in levelnodes for i in farptr[idx]:(farptr[idx + 1] - 1)]
        isempty(faridcs) && continue
        levelrowptr = [
            1
            cumsum([farptr[idx + 1] - farptr[idx] for idx in levelnodes]) .+ 1
        ]
        push!(
            farinteractionmatrix,
            VariableBlockCompressedRowStorage{
                eltype(kernelmatrix),eltype(blocks),Int,typeof(scheduler)
            }(
                blocks[faridcs],
                levelrowptr,
                first.(farvalues[faridcs]),
                first.(values[levelnodes]),
                (length(testspace), length(trialspace)),
                scheduler,
            ),
        )
    end

    return farinteractionmatrix
end
