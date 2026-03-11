
function farinteractions(tree; args...)
    return error("Needs to be implemented for $(typeof(tree))")
end

function farinteractions_consecutive(tree; args...)
    return error("Needs to be implemented for $(typeof(tree))")
end
#=  
    Wrapper for LowRankMatrix which will preserve the ordering of the original matrix.
    values and farvalues are vectors of the indices of respectively rows and columns corresponding to this block.
    dim is the size of the original matrix
=#
struct FarIntractionMatrix{T} <: AbstractMatrix{T}
    block::LowRankMatrix{T}
    values::Vector{Int}
    farvalues::Vector{Int}
    dim::Tuple{Int,Int}
end

Base.size(A::FarIntractionMatrix) = A.dim

function Base.getindex(A::FarIntractionMatrix{T}, i::Int, j::Int) where {T}
    checkbounds(A, i, j)
    
    rloc = findfirst(==(i), A.values)
    rloc === nothing && return zero(T)
    
    cloc = findfirst(==(j), A.farvalues)
    cloc === nothing && return zero(T)
    
    blk = A.block
    return dot(view(blk.U, rloc, :), view(blk.V, :, cloc))
end

function Base.Matrix(A::FarIntractionMatrix{T}) where {T}
    mat = zeros(T, A.dim)
    mat[A.values, A.farvalues] .+= Matrix(A.block)
    return mat
end

#=
    Here the smaller far blocks from the consecutive path are kept,
    because joining them led to 'Full rank rectangular matrix' warning and failure of the tolerance test.
=#
function assemblefars(
    operator,
    testspace,
    trialspace,
    tree;
    compressor=ACA(),
    isnear=isnear(),
    matrixdata=defaultfarmatrixdata(operator, testspace, trialspace),
    maxrank=50,
    scheduler=SerialScheduler(),
)
    kernelmatrix = AbstractKernelMatrix(
        operator, testspace, trialspace; matrixdata=matrixdata
    )
    valptr, values, farvalues = farinteractions(tree; isnear=isnear)

    farinteractionmatrices = Vector{FarIntractionMatrix{eltype(kernelmatrix)}}(undef, length(farvalues))
    colbuffer = zeros(eltype(kernelmatrix), length(testspace), maxrank)
    farinteractionmatrix = VariableBlockCompressedRowStorage[]
    farvalues == [] ? buffersize = 0 : buffersize = maximum(length.(farvalues))
    for level in levels(testtree(tree))
        fnodes = collect(LevelIterator(testtree(tree), level))
        @tasks for node in fnodes
            @set scheduler = scheduler
            @local rowbuffer = zeros(eltype(kernelmatrix), maxrank, buffersize)
            for faridx in valptr[node]:(valptr[node + 1] - 1)
                npivots = compressor(
                    kernelmatrix,
                    view(colbuffer, values[node], 1:maxrank),
                    rowbuffer,
                    maxrank;
                    rowidcs=values[node],
                    colidcs=farvalues[faridx],
                )
                farinteractionmatrices[faridx] = FarIntractionMatrix(
                    LowRankMatrix(
                        colbuffer[values[node], 1:npivots],
                        rowbuffer[1:npivots, 1:length(farvalues[faridx])],
                    ),
                    values[node],
                    farvalues[faridx],
                    size(kernelmatrix)
                )
                colbuffer[values[node], 1:npivots] .= eltype(kernelmatrix)(0)
                rowbuffer[1:npivots, 1:length(farvalues[faridx])] .= eltype(kernelmatrix)(0)
            end
        end
    end

    return farinteractionmatrices
end


function assemblefars_consecutive(
    operator,
    testspace,
    trialspace,
    tree;
    compressor=ACA(),
    isnear=isnear(),
    matrixdata=defaultfarmatrixdata(operator, testspace, trialspace),
    maxrank=50,
    scheduler=SerialScheduler(),
)
    kernelmatrix = AbstractKernelMatrix(
        operator, testspace, trialspace; matrixdata=matrixdata
    )
    valptr, values, farvalues = farinteractions_consecutive(tree; isnear=isnear)

    blocks = Vector{LowRankMatrix{eltype(kernelmatrix)}}(undef, length(farvalues))
    colbuffer = zeros(eltype(kernelmatrix), length(testspace), maxrank)
    farinteractionmatrix = VariableBlockCompressedRowStorage[]
    farvalues == [] ? buffersize = 0 : buffersize = maximum(length.(farvalues))
    for level in levels(testtree(tree))
        fnodes = collect(LevelIterator(testtree(tree), level))
        @tasks for node in fnodes
            @set scheduler = scheduler
            @local rowbuffer = zeros(eltype(kernelmatrix), maxrank, buffersize)
            for faridx in valptr[node]:(valptr[node + 1] - 1)
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

        faridcs = [i for idx in fnodes for i in valptr[idx]:(valptr[idx + 1] - 1)]
        isempty(faridcs) && continue
        levelrowptr = [1; cumsum([valptr[idx + 1] - valptr[idx] for idx in fnodes]) .+ 1]
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
