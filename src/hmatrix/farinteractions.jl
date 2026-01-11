
function farinteractions(tree; args...)
    return error("Needs to be implemented for $(typeof(tree))")
end

function assemblefars(
    operator,
    testspace,
    trialspace,
    tree;
    compressor=ACA(; tol=1e-4),
    isnear=isnear(),
    quadstrat=defaultfarquadstrat(operator, testspace, trialspace),
    maxrank=50,
    ntasks=Threads.nthreads(),
)
    kernelmatrix = KernelMatrix(operator, testspace, trialspace; quadstrat=quadstrat)
    vals, lengths, farvals, farlengths = farinteractions(tree; isnear=isnear)

    colbuffer = zeros(scalartype(operator), length(testspace), maxrank)
    farinteractionmatrix = Vector{VariableBlockCompressedRowStorage}(undef, length(fars))
    for level in eachindex(vals)
        blocks = Vector{Vector{LowRankBlocks{scalartype(operator)}}}(
            undef, length(vals[level])
        )
        @tasks for validx in eachindex(vals[level])
            @set ntasks = ntasks
            @local rowbuffer = zeros(scalartype(operator), maxrank, length(trialspace))
            localblocks = Vector{LowRankBlocks{scalartype(operator)}}(
                undef, length(farvals[level][validx])
            )
            rows = fars[level][validx]:(fars[level][validx] + lengths[level][validx] - 1)
            for (far, faridx) in enumerate(farvals[level][validx])
                cols = far:(far + farlengths[level][validx][faridx] - 1)
                npivots = compressor(
                    kernelmatrix,
                    view(colbuffer, rows, 1:maxrank),
                    rowbuffer,
                    maxrank;
                    rowidcs=rows,
                    colidcs=cols,
                )
                localblocks[faridx] = LowRankMatrix(
                    colbuffer[rows, 1:npivots], rowbuffer[1:npivots, 1:length(cols)]
                )
            end
            blocks[validx] = localblocks
        end
        farinteractionmatrix[level] = VariableBlockCompressedRowStorage(
            blocks, vals[level], farvals[level], (length(testspace), length(trialspace))
        )
    end

    return farinteractionmatrix
end
