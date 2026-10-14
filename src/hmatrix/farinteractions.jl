"""
    farinteractions(tree; isnear=isnear(), spaceordering=PreserveSpaceOrder())
    farinteractions(tree, isnear, spaceordering)

Extract admissible far-field cluster pairs. The keyword entry point forwards to
positional backend methods, dispatching on the admissibility functor and space
ordering.

Return `(values, farptr, farvalues)`, grouping admissible column clusters by row
cluster. Use `symmetricfarinteractions` to visit each unordered pair once.

`PreserveSpaceOrder()` returns indices in the original space order.
`PermuteSpaceInPlace()` returns indices for spaces already permuted to tree order.
The traversal does not permute spaces itself.
"""
function farinteractions(
    tree; isnear=isnear(), spaceordering::SpaceOrderingStyle=PreserveSpaceOrder(), kwargs...
)
    return farinteractions(tree, IsNearFunctor(isnear), spaceordering; kwargs...)
end

function farinteractions(
    tree, isnear::IsNearFunctor, spaceordering::SpaceOrderingStyle; kwargs...
)
    return error(
        "Far interactions are not implemented for $(typeof(tree)), " *
        "$(typeof(isnear)), $(typeof(spaceordering))",
    )
end

"""
    symmetricfarinteractions(tree; isnear=isnear(), spaceordering=PreserveSpaceOrder())
    symmetricfarinteractions(tree, isnear, spaceordering)

Return `(values, farptr, farvalues, diagonalflags)`, grouping unordered admissible
pairs by row cluster. For a row node `i`, `farptr[i]:(farptr[i + 1] - 1)` indexes
its column clusters in `farvalues`; `diagonalflags[j]` marks a self block.
Only the test cluster tree is used for both axes. The admissibility predicate
must be transpose-symmetric.

`PreserveSpaceOrder()` returns indices in the original space order.
`PermuteSpaceInPlace()` returns indices for spaces already permuted to tree order.
The keyword entry point forwards to positional backend methods for each ordering.
Both orderings support traversal, although symmetric matrix assembly currently
supports only `PreserveSpaceOrder()`.
"""
function symmetricfarinteractions(
    tree; isnear=isnear(), spaceordering::SpaceOrderingStyle=PreserveSpaceOrder(), kwargs...
)
    return symmetricfarinteractions(tree, IsNearFunctor(isnear), spaceordering; kwargs...)
end

function symmetricfarinteractions(
    tree, isnear::IsNearFunctor, spaceordering::SpaceOrderingStyle; kwargs...
)
    return error(
        "Symmetric far interactions are not implemented for $(typeof(tree)), " *
        "$(typeof(isnear)), $(typeof(spaceordering))",
    )
end

"""
    assemblefars(operator, testspace, trialspace, tree, ::PreserveSpaceOrder; kwargs...)

Assemble far-field blocks without reordering test and trial spaces.

Compresses admissible, well-separated cluster pairs using ACA-style algorithms,
maintaining the original space ordering. Produces a collection of level-by-level
block-sparse matrices.

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
  - `verbose`: enable progress output (default `true`)

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
    spaceordering::PreserveSpaceOrder;
    compressor=ACA(),
    isnear=isnear(),
    matrixdata=defaultfarmatrixdata(operator, testspace, trialspace),
    maxrank=50,
    scheduler=SerialScheduler(),
    verbose::Bool=true,
)
    kernelmatrix = AbstractKernelMatrix(
        operator, testspace, trialspace; matrixdata=matrixdata
    )
    values, farptr, farvalues = farinteractions(
        tree; spaceordering=spaceordering, isnear=isnear
    )

    (farptr[end] == 1) && return [
        BlockSparseMatrix(
            Matrix{eltype(kernelmatrix)}[],
            Vector{Int}[],
            Vector{Int}[],
            size(kernelmatrix),
        ),
    ]

    total_work = sum(
        length(values[i]) * length(farvalues[j]) for i in eachindex(values) for
        j in farptr[i]:(farptr[i + 1] - 1);
        init=0,
    )
    progress = Progress(
        total_work; desc="Assembling far interactions: ", showspeed=true, enabled=verbose
    )

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
                localmatrix = localkernelmatrix(
                    kernelmatrix, values[node], farvalues[faridx]
                )
                npivots = localcompressor(
                    localmatrix,
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
                next!(progress; step=length(values[node]) * length(farvalues[faridx]))
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
    finish!(progress)

    return farinteractionmatrix
end

"""
    assemblefars(operator, testspace, trialspace, tree, ::PermuteSpaceInPlace; kwargs...)

Assemble far-field blocks with tree-aligned space reordering.

Compresses admissible, well-separated cluster pairs using ACA-style algorithms,
with both spaces permuted to align with the hierarchical tree structure. Produces
a specialized storage format optimized for the reordered layout.

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
  - `verbose`: enable progress output (default `true`)

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
    spaceordering::PermuteSpaceInPlace;
    compressor=ACA(),
    isnear=isnear(),
    matrixdata=defaultfarmatrixdata(operator, testspace, trialspace),
    maxrank=50,
    scheduler=SerialScheduler(),
    verbose::Bool=true,
)
    kernelmatrix = AbstractKernelMatrix(
        operator, testspace, trialspace; matrixdata=matrixdata
    )
    values, farptr, farvalues = farinteractions(
        tree; spaceordering=spaceordering, isnear=isnear
    )

    total_work = sum(
        length(values[i]) * length(farvalues[j]) for i in eachindex(values) for
        j in farptr[i]:(farptr[i + 1] - 1);
        init=0,
    )
    progress = Progress(
        total_work; desc="Assembling far interactions: ", showspeed=true, enabled=verbose
    )

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
                localmatrix = localkernelmatrix(
                    kernelmatrix, values[node], farvalues[faridx]
                )
                npivots = localcompressor(
                    localmatrix,
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
                next!(progress; step=length(values[node]) * length(farvalues[faridx]))
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
    finish!(progress)

    return farinteractionmatrix
end

"""
    assemblefars(operator, space, tree, spaceordering; kwargs...)

Compress each unordered far-field pair once and use its transpose for the
opposite matrix block. The admissibility predicate and operator must be
transpose-symmetric.

Only `PreserveSpaceOrder()` is supported; symmetric VBCRS storage is required
before `PermuteSpaceInPlace()` can be used.
"""
function assemblefars(
    operator,
    space,
    tree,
    spaceordering::SpaceOrderingStyle;
    compressor=ACA(),
    isnear=isnear(),
    matrixdata=defaultfarmatrixdata(operator, space, space),
    maxrank=50,
    scheduler=SerialScheduler(),
    mvpscheduler=scheduler,
    verbose::Bool=true,
)
    _check_symmetric_ordering(spaceordering)
    kernelmatrix = AbstractKernelMatrix(operator, space, space; matrixdata=matrixdata)
    values, farptr, farvalues, diagonalflags = symmetricfarinteractions(
        tree; spaceordering=spaceordering, isnear=isnear
    )
    T = eltype(kernelmatrix)
    farinteractionmatrix = BlockSparseMatrices.AbstractBlockMatrix{T}[]
    farptr[end] == 1 && return farinteractionmatrix

    total_work = sum(
        length(values[i]) * length(farvalues[j]) for i in eachindex(values) for
        j in farptr[i]:(farptr[i + 1] - 1);
        init=0,
    )
    progress = Progress(
        total_work;
        desc="Assembling symmetric far interactions: ",
        showspeed=true,
        enabled=verbose,
    )

    blocks = Vector{LowRankMatrix{T}}(undef, length(farvalues))
    colbuffer = zeros(T, length(space), maxrank)
    for level in levels(testtree(tree))
        levelnodes = collect(LevelIterator(testtree(tree), level))
        rbsize, cbsize = buffersize(values, farptr, farvalues, levelnodes)
        cbsize == 0 && continue
        @tasks for node in levelnodes
            @set scheduler = scheduler
            @local begin
                rowbuffer = zeros(T, maxrank, cbsize)
                localcompressor = compressor(kernelmatrix, rbsize, cbsize, maxrank)
            end
            for faridx in farptr[node]:(farptr[node + 1] - 1)
                localmatrix = localkernelmatrix(
                    kernelmatrix, values[node], farvalues[faridx]
                )
                npivots = localcompressor(
                    localmatrix,
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
                colbuffer[values[node], 1:npivots] .= zero(T)
                rowbuffer[1:npivots, 1:length(farvalues[faridx])] .= zero(T)
                next!(progress; step=length(values[node]) * length(farvalues[faridx]))
            end
        end

        levelrows, levelcols, levelidcs = blockvalues(values, farptr, farvalues, levelnodes)
        diagonals = Matrix{T}[]
        diagindices = Vector{Int}[]
        offdiagonals = LowRankMatrix{T}[]
        offrows = Vector{Int}[]
        offcols = Vector{Int}[]
        for i in eachindex(levelidcs)
            faridx = levelidcs[i]
            if diagonalflags[faridx]
                block = Matrix(blocks[faridx])
                for j in axes(block, 2), k in (j + 1):size(block, 1)
                    block[k, j] = block[j, k]
                end
                push!(diagonals, block)
                push!(diagindices, levelrows[i])
            else
                push!(offdiagonals, blocks[faridx])
                push!(offrows, levelrows[i])
                push!(offcols, levelcols[i])
            end
        end

        if isempty(offdiagonals)
            push!(
                farinteractionmatrix,
                BlockSparseMatrix(
                    diagonals,
                    diagindices,
                    diagindices,
                    size(kernelmatrix);
                    scheduler=mvpscheduler,
                ),
            )
        elseif mvpscheduler isa SerialScheduler
            push!(
                farinteractionmatrix,
                _serialsymmetricblockmatrix(
                    diagonals,
                    diagindices,
                    offdiagonals,
                    offrows,
                    offcols,
                    size(kernelmatrix),
                    mvpscheduler,
                ),
            )
        else
            if isempty(diagonals)
                push!(diagonals, zeros(T, 1, 1))
                push!(diagindices, [1])
            end
            push!(
                farinteractionmatrix,
                SymmetricBlockMatrix(
                    diagonals,
                    diagindices,
                    offdiagonals,
                    offrows,
                    offcols,
                    size(kernelmatrix);
                    scheduler=mvpscheduler,
                ),
            )
        end
    end
    finish!(progress)
    return farinteractionmatrix
end
