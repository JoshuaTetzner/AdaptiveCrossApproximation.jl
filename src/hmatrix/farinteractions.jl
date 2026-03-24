
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
    spaceordering::PreserveSpaceOrder;
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

    blocks = Vector{LowRankMatrix{eltype(kernelmatrix)}}(undef, length(farvalues))
    colbuffer = zeros(eltype(kernelmatrix), length(testspace), maxrank)
    farinteractionmatrix = VariableBlockCompressedRowStorage[]
    farvalues == [] ? buffersize = 0 : buffersize = maximum(length.(farvalues))
    for level in levels(testtree(tree))
        levelnodes = collect(LevelIterator(testtree(tree), level))
        bsize = buffersize(farptr, farvalues, levelnodes)
        @tasks for node in levelnodes
            @set scheduler = scheduler
            @local rowbuffer = zeros(eltype(kernelmatrix), maxrank, bsize)
            for faridx in farptr[node]:(farptr[node + 1] - 1)
                npivots = compressor(
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

        isempty(faridcs) && continue
    end

    return farinteractionmatrix
end

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
)
    kernelmatrix = AbstractKernelMatrix(
        operator, testspace, trialspace; matrixdata=matrixdata
    )
    values, farptr, farvalues = farinteractions_consecutive(tree; isnear=isnear)

    blocks = Vector{LowRankMatrix{eltype(kernelmatrix)}}(undef, length(farvalues))
    colbuffer = zeros(eltype(kernelmatrix), length(testspace), maxrank)
    farinteractionmatrix = VariableBlockCompressedRowStorage[]
    for level in levels(testtree(tree))
        fnodes = collect(LevelIterator(testtree(tree), level))
        bsize = buffersize(farptr, farvalues, fnodes)
        @tasks for node in fnodes
            @set scheduler = scheduler
            @local rowbuffer = zeros(eltype(kernelmatrix), maxrank, bsize)
            for faridx in farptr[node]:(farptr[node + 1] - 1)
                npivots = compressor(
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

        faridcs = [i for idx in fnodes for i in farptr[idx]:(farptr[idx + 1] - 1)]
        isempty(faridcs) && continue
        levelrowptr = [1; cumsum([farptr[idx + 1] - farptr[idx] for idx in fnodes]) .+ 1]
        push!(
            farinteractionmatrix,
            VariableBlockCompressedRowStorage{
                eltype(kernelmatrix),eltype(blocks),Int,typeof(scheduler)
            }(
                blocks[faridcs],
                levelrowptr,
                first.(farvalues[faridcs]),
                first.(values[fnodes]),
                (length(testspace), length(trialspace)),
                scheduler,
            ),
        )
    end

    return farinteractionmatrix
end
