struct IsNearFunctor{F}
    η::F
end

function isnear(; η=1.0)
    return IsNearFunctor{F}(η)
end

function nearinteractions(tree; args...)
    return error("Needs to be implemented for $(typeof(tree))")
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
    tree;
    isnear=isnear(),
    ntasks=Threads.nthreads(),
    quadstrat=defaultquadstrat(operator, testspace, trialspace),
)
    nearmatrix = AbstractKernelMatrix(operator, testspace, trialspace; quadstrat=quadstrat)
    values, lenvalues, nearvalues, lennearvalues = nearinteractions(tree; isnear=isnear)
    blocks = zeros.(eltype(nearmatrix), lenvalues, sum(lennearvalues))
    viewblocks = Vector{
        Vector{
            SubArray{
                Float64,
                2,
                Matrix{scalartype(operator)},
                Tuple{UnitRange{Int64},UnitRange{Int64}},
                false,
            },
        },
    }(
        undef, length(blocks)
    )
    @time @tasks for i in eachindex(blocks)
        @set ntasks = ntasks
        nearmatrix(blocks[i], values[i][1], nearvalues[i])
        viewblocks[i] = splitblock(blocks[i], lennearvalues[i])
    end

    return VariableBlockCompressedRowStorage(
        collect(Iterators.flatten(viewblocks)),
        collect(Iterators.flatten(values)),
        collect(Iterators.flatten(nearvalues)),
        size(nearmatrix),
    )
end
