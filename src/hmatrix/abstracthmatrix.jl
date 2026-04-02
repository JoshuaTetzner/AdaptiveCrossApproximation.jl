using BlockSparseMatrices
using LinearAlgebra
using LinearMaps
using OhMyThreads

defaultmatrixdata(operator, testspace, trialspace) = nothing
defaultfarmatrixdata(operator, testspace, trialspace) = nothing
defaultcompressor(operator, testspace, trialspace) = ACA(; tol=1e-4)

# kernelmatrix code
scalartype(operator) = error("Not implemented for $(typeof(operator))")
permute(space, perm) = permute!(copy(space), perm)

# tree code
abstract type AbstractTree end
struct H2Tree <: AbstractTree end

function _tree(::AbstractTree, args...; kwargs...)
    return error("Please load H2Trees.jl or your custom tree implementation.")
end

testtree(tree) = error("Requires implementation for $(typeof(tree))")
trialtree(tree) = error("Requires implementation for $(typeof(tree))")
levels(tree) = error("Requires implementation for $(typeof(tree))")
LevelIterator(tree, level) = error("Requires implementation for $(typeof(tree))")
permutation(tree) = error("Requires implementation for $(typeof(tree))")

abstract type SpaceOrderingStyle end
struct PermuteSpaceInPlace <: SpaceOrderingStyle end
function (::PermuteSpaceInPlace)(tree, testspace, trialspace)
    testperm = permutation(testtree(tree))
    permute!(testspace, testperm)

    if testspace === trialspace && testtree(tree) === trialtree(tree)
        return nothing
    elseif !(testspace === trialspace) && !(testtree(tree) === trialtree(tree))
        trialperm = permutation(trialtree(tree))
        permute!(trialspace, trialperm)
        return nothing
    else
        @warn "Risky territory: Permuting trialtree not trialspace."
        trialperm = permutation(trialtree(tree))
        return nothing
    end
end
struct PreserveSpaceOrder <: SpaceOrderingStyle end
function (::PreserveSpaceOrder)(tree, testspace, trialspace)
    return nothing
end

include("hmatrix.jl")
include("nearinteractions.jl")
include("skeleton.jl")
include("farinteractions.jl")

function HMatrix(
    operator,
    testspace,
    trialspace,
    tree;
    tol=1e-4,
    compressor=ACA(; tol=tol),
    isnear=isnear(),
    maxrank=40,
    spaceordering::SpaceOrderingStyle=PermuteSpaceInPlace(),
    nearmatrixdata=defaultmatrixdata(operator, testspace, trialspace),
    farmatrixdata=defaultfarmatrixdata(operator, testspace, trialspace),
    scheduler=DynamicScheduler(),
)
    spaceordering(tree, testspace, trialspace)

    nears = assemblenears(
        operator,
        testspace,
        trialspace,
        tree,
        spaceordering;
        isnear=isnear,
        matrixdata=nearmatrixdata,
        scheduler=scheduler,
    )

    fars = assemblefars(
        operator,
        testspace,
        trialspace,
        tree,
        spaceordering;
        maxrank=maxrank,
        compressor=compressor,
        isnear=isnear,
        matrixdata=farmatrixdata,
        scheduler=scheduler,
    )

    return HMatrix{eltype(nears)}(nears, fars, (length(testspace), length(trialspace)))
end

function HMatrix(
    operator,
    space,
    tree;
    isnear=isnear(),
    compressor=ACA(; tol=1e-4),
    permutation=true,
    nearquadstrat=defaultmatrixdata(operator, space, space),
    farquadstrat=defaultfarmatrixdata(operator, space, space),
    ntasks=Threads.nthreads(),
)
    return error("Symmetric version not implemented yet")
end

function farmatrix(hmat::HMatrix)
    blocks = Matrix{eltype(hmat)}[]
    nears = BlockSparseMatrix(blocks, Vector{Int}[], Vector{Int}[], hmat.dim)

    return HMatrix{eltype(hmat)}(nears, hmat.farinteractions, hmat.dim)
end

function nearmatrix(hmat::HMatrix)
    return hmat.nearinteractions
end

function storage(hmat::HMatrix)
    refsize = size(hmat, 1) * size(hmat, 2) * sizeof(eltype(hmat))
    matsize = 0
    for blk in hmat.nearinteractions.blocks
        matsize += length(blk)
    end
    for farmat in hmat.farinteractions
        for blk in farmat.blocks
            matsize += length(blk)
        end
    end
    println("storage: ", matsize * sizeof(eltype(hmat)) * 10^-9, " GB")
    println("summary size: ", Base.summarysize(hmat) * 10^-9, " GB")
    println("compression ratio: ", (matsize * sizeof(eltype(hmat))) / refsize)
    return matsize * sizeof(eltype(hmat)) * 10^-9
end
