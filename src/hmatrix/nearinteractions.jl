struct IsNearFunctor{F}
    η::F
end

function isnear(η::Real=Float64(1.0))
    return IsNearFunctor{typeof(η)}(η)
end

function nearinteractions(tree; args...)
    return error("Needs to be implemented for $(typeof(tree))")
end
function nearinteractions_consecutive(tree; args...)
    return error("Needs to be implemented for $(typeof(tree))")
end

function assemblenears(
    operator,
    testspace,
    trialspace,
    tree,
    ::PreserveSpaceOrder;
    isnear=isnear(),
    scheduler=SerialScheduler(),
    matrixdata=defaultmatrixdata(operator, testspace, trialspace),
)
    nearmatrix = AbstractKernelMatrix(
        operator, testspace, trialspace; matrixdata=matrixdata
    )
    values, nearvalues = nearinteractions(tree; isnear=isnear)

    isempty(values) && return BlockSparseMatrix(
        Matrix{eltype(nearmatrix)}[], Vector{Int}[], Vector{Int}[], size(nearmatrix)
    )

    blocks = Vector{Matrix{eltype(nearmatrix)}}(undef, length(values))
    @tasks for i in eachindex(blocks)
        @set scheduler = scheduler
        blk = zeros(eltype(nearmatrix), length(values[i]), length(nearvalues[i]))
        nearmatrix(blk, values[i], nearvalues[i])
        blocks[i] = blk
    end

    nears = BlockSparseMatrix(
        blocks, values, nearvalues, size(nearmatrix); scheduler=scheduler
    )

    return nears
end

function splitblock(block::Matrix{T}, lens::Vector{Int}) where {T}
    return [
        view(block, 1:size(block, 1), sum(lens[1:(i - 1)]) .+ (1:lens[i])) for
        i in eachindex(lens)
    ]
end

function assemblenears(
    operator,
    testspace,
    trialspace,
    tree,
    ::PermuteSpaceInPlace;
    isnear=isnear(),
    scheduler=SerialScheduler(),
    matrixdata=defaultmatrixdata(operator, testspace, trialspace),
)
    nearmatrix = AbstractKernelMatrix(
        operator, testspace, trialspace; matrixdata=matrixdata
    )
    values, nearvalues = nearinteractions_consecutive(tree; isnear=isnear)
    blocks = zeros.(
        eltype(nearmatrix), length.(values), [sum(length.(n)) for n in nearvalues]
    )
    # There should be a prettier not hardcoded way to do this, but it works for now
    viewblocks = Vector{
        Vector{
            SubArray{
                eltype(nearmatrix),
                2,
                Matrix{eltype(nearmatrix)},
                Tuple{UnitRange{Int},UnitRange{Int}},
                false,
            },
        },
    }(
        undef, length(blocks)
    )
    @tasks for i in eachindex(blocks)
        @set scheduler = scheduler
        nearmatrix(blocks[i], values[i], Iterators.flatten(nearvalues[i]))
        viewblocks[i] = splitblock(blocks[i], length.(nearvalues[i]))
    end
    mat = VariableBlockCompressedRowStorage{
        eltype(nearmatrix),eltype(Iterators.flatten(viewblocks)),Int,typeof(scheduler)
    }(
        collect(Iterators.flatten(viewblocks)),
        [1; cumsum(length.(nearvalues)) .+ 1],
        first.(Iterators.flatten(nearvalues)),
        first.(values),
        size(nearmatrix),
        scheduler,
    )
    return mat
end
