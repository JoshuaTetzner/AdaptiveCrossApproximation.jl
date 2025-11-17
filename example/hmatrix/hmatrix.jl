using H2Trees
using OhMyThreads
using BlockSparseMatrices
using AdaptiveCrossApproximation

struct HMatrix{I,K,NearInteractionType,FarInteractionType} <: LinearMaps.LinearMap{K}
    nearinteractions::NearInteractionType
    farinteractions::FarInteractionType
    dim::Tuple{I,I}
    ntasks::I
    function HMatrix{I,K}(nearinteractions, farinteractions, dim, ntasks) where {I,K}
        return new{I,K,typeof(nearinteractions),typeof(farinteractions)}(
            nearinteractions, farinteractions, dim, ntasks
        )
    end
end

function LinearMaps._unsafe_mul!(
    y::AbstractVector, A::M, x::AbstractVector
) where {M<:HMatrix}
    fill!(y, zero(eltype(y)))

    mul!(y, A.nearinteractions, x)
    for level in A.farinteractions
        @tasks for interactions in level
            @set ntasks = A.ntasks
            for lrb in interactions
                y[lrb.τ] += lrb.M * x[lrb.σ]
            end
        end
    end

    return y
end

function Base.size(A::HMatrix, dim=nothing)
    if dim === nothing
        return (A.dim[1], A.dim[2])
    elseif dim == 1
        return A.dim[1]
    elseif dim == 2
        return A.dim[2]
    end
end

struct IsNearFunctor{F}
    η::F
end

function isnear(η::F) where {F}
    return IsNearFunctor{F}(η)
end

function (isnear::IsNearFunctor{F})(
    treea::H2Trees.TwoNTree, treeb::H2Trees.TwoNTree, nodea::Int, nodeb::Int
) where {F}
    ths = H2Trees.halfsize(treea, nodea) * sqrt(3)
    shs = H2Trees.halfsize(treeb, nodeb) * sqrt(3)
    dist = norm(H2Trees.center(treea, nodea) - H2Trees.center(treeb, nodeb)) - (ths + shs)

    (2 * max(ths, shs) <= isnear.η * max(dist, 0.0)) ? (return false) : (return true)
end

function (isnear::IsNearFunctor{F})(
    treea::H2Trees.BoundingBallTree, treeb::H2Trees.BoundingBallTree, nodea::Int, nodeb::Int
) where {F}
    ths = H2Trees.radius(treea, nodea) * sqrt(3)
    shs = H2Trees.radius(treeb, nodeb) * sqrt(3)
    dist = norm(H2Trees.center(treea, nodea) - H2Trees.center(treeb, nodeb)) - (ths + shs)

    (2 * max(ths, shs) <= isnear.η * max(dist, 0.0)) ? (return false) : (return true)
end

function HMatrix(
    operator,
    testspace,
    trialspace,
    tree::BlockTree;
    isnear=isnear(1.0),
    nearquadstrat=BEAST.defaultquadstrat(operator, testspace, trialspace),
    farquadstrat=BEAST.DoubleNumQStrat(2, 3),
    compressor=ACA(),
    maxrank=30,
    ntasks=Threads.nthreads(),
)
    lk = Threads.SpinLock()
    T = scalartype(operator)

    nearmatrix = AbstractKernel(operator, testspace, trialspace; quadstrat=nearquadstrat)
    values, nearvalues = H2Trees.nearinteractions(
        tree; isnear=isnear, extractselfvalues=false
    )

    println("nearinteractions")
    blocks = Vector{Matrix{T}}(undef, length(values))
    @time @tasks for i in eachindex(values)
        @set ntasks = ntasks
        blk = zeros(T, length(values[i]), length(nearvalues[i]))
        nearmatrix(blk, values[i], nearvalues[i])
        blocks[i] = blk
    end
    nearinteractions = BlockSparseMatrix(
        blocks, values, nearvalues, (length(testspace), length(trialspace))
    )

    # storing the far interactions as vector of Dicts, each Dict corresponding to
    # one level, each Dict contains the nodes on the level.
    # Allows easy multithreading in MV
    farmatrix = AbstractKernel(operator, testspace, trialspace; quadstrat=farquadstrat)
    iterator = H2Trees.WellSeparatedIterator(; isnear=(tree) -> isnear)(tree)
    farinteractions = Vector{Vector{MatrixBlock{Int,T,LowRankMatrix{T}}}}[]

    colbuffer = zeros(T, length(testspace), maxrank)
    rowbuffer = Channel{Matrix{T}}(ntasks)
    for _ in 1:ntasks
        put!(rowbuffer, zeros(T, maxrank, length(trialspace)))
    end

    println("farinteractions")
    @time for level in H2Trees.levels(tree.testcluster)
        leveldfarblocks = Vector{MatrixBlock{Int,T,LowRankMatrix{T}}}[]
        @tasks for t in collect(H2Trees.LevelIterator(tree.testcluster, level))
            farblocks = MatrixBlock{Int,T,LowRankMatrix{T}}[]
            @set ntasks = ntasks
            localrowbuffer = take!(rowbuffer)
            for s in iterator(tree.trialcluster, tree.testcluster, t)
                tvals = H2Trees.values(tree.testcluster, t)
                svals = H2Trees.values(tree.trialcluster, s)
                rows = zeros(Int, maxrank)
                cols = zeros(Int, maxrank)

                npivots = compressor(
                    farmatrix,
                    view(colbuffer, tvals, 1:maxrank),
                    view(localrowbuffer, 1:maxrank, svals),
                    min(maxrank, min(length(tvals), length(svals)));
                    rows=rows,
                    cols=cols,
                    rowidcs=tvals,
                    colidcs=svals,
                )
                lrb = MatrixBlock{Int,T,LowRankMatrix{T}}(
                    LowRankMatrix(
                        colbuffer[tvals, 1:npivots], localrowbuffer[1:npivots, svals]
                    ),
                    tvals,
                    svals,
                )
                push!(farblocks, lrb)
                colbuffer[tvals, 1:npivots] .= T(0)
                localrowbuffer[1:npivots, svals] .= T(0)
            end
            put!(rowbuffer, localrowbuffer)
            lock(lk) do
                push!(leveldfarblocks, farblocks)
            end
        end
        push!(farinteractions, leveldfarblocks)
    end

    return HMatrix{Int,T}(
        nearinteractions, farinteractions, (length(testspace), length(trialspace)), ntasks
    )
end
