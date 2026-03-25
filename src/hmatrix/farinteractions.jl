
function farinteractions(tree; args...)
    return error("Needs to be implemented for $(typeof(tree))")
end

function farinteractions_consecutive(tree; args...)
    return error("Needs to be implemented for $(typeof(tree))")
end

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
                blocks[faridx] = LowRankMatrix(
                    colbuffer[values[node], 1:npivots],
                    rowbuffer[1:npivots, 1:length(farvalues[faridx])],
                )
                colbuffer[values[node], 1:npivots] .= eltype(kernelmatrix)(0)
                rowbuffer[1:npivots, 1:length(farvalues[faridx])] .= eltype(kernelmatrix)(0)
            end
        end
        levelvals, levelfarvals = blockvalues(values, farptr, farvalues, levelnodes)
        push!(
            farinteractionmatrix,
            BlockSparseMatrix(
                blocks, levelvals, levelfarvals, size(kernelmatrix); scheduler=scheduler
            ),
        )
    end

    return farinteractionmatrix
end

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
