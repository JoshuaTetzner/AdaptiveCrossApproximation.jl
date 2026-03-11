using BlockSparseMatrices
using LinearAlgebra
using LinearMaps
using OhMyThreads
using SparseArrays # to store near interaactions

function defaultmatrixdata(operator, testspace, trialspace) end
function defaultfarmatrixdata(operator, testspace, trialspace) end

function defaultcompressor(operator, testspace, trialspace)
    return ACA(; tol=1e-4)
end
scalartype(operator) = error("Not implemented for $(typeof(operator))")

testtree(tree) = error("Requiers implementation for $(typeof(tree))")
trialtree(tree) = error("Requiers implementation for $(typeof(tree))")
levels(tree) = error("Requiers implementationfor $(typeof(tree))")
LevelIterator(tree, level) = error("Requiers implementation for $(typeof(tree))")

permutation(tree) = error("Requiers implementation for $(typeof(tree))")
permute(space, perm) = permute!(copy(space), perm)

abstract type SpaceOrderingStyle end
struct PermuteSpaceInPlace <: SpaceOrderingStyle end
struct PreserveSpaceOrder <: SpaceOrderingStyle end

include("hmatrix.jl")
include("permutedhmatrix.jl")
include("nearinteractions.jl")
include("skeleton.jl")
include("farinteractions.jl")

function HMatrix(
    operator,
    testspace,
    trialspace,
    tree;
    space_ordering::SpaceOrderingStyle=PermuteSpaceInPlace(),
    tol=1e-4,
    compressor=ACA(; tol=tol),
    isnear=isnear(),
    nearmatrixdata=defaultmatrixdata(operator, testspace, trialspace),
    farmatrixdata=defaultfarmatrixdata(operator, testspace, trialspace),
    scheduler=DynamicScheduler(),
)
    return _hmatrix(
            operator,
            testspace,
            trialspace,
            tree,
            space_ordering;
            tol=tol,
            compressor=compressor,
            isnear=isnear,
            nearmatrixdata=nearmatrixdata,
            farmatrixdata=farmatrixdata,
            scheduler=scheduler,
        )
end

function _hmatrix(
    operator,
    testspace,
    trialspace,
    tree,
    space_ordering::PreserveSpaceOrder;
    tol=1e-4,
    compressor=ACA(; tol=tol),
    isnear=isnear(),
    nearmatrixdata=defaultmatrixdata(operator, testspace, trialspace),
    farmatrixdata=defaultfarmatrixdata(operator, testspace, trialspace),
    scheduler=DynamicScheduler(),
)

    nears = assemblenears(
        operator,
        testspace,
        trialspace,
        tree;
        isnear=isnear,
        matrixdata=nearmatrixdata,
        scheduler=scheduler,
    )

    fars = assemblefars(
        operator,
        testspace,
        trialspace,
        tree;
        compressor=compressor,
        isnear=isnear,
        matrixdata=farmatrixdata,
        scheduler=scheduler,
    )
    print(typeof(fars), "\n")

    return HMatrix{eltype(nears)}(nears, fars, (length(testspace), length(trialspace)))
end

function _hmatrix(
    operator,
    testspace,
    trialspace,
    tree,
    space_ordering::PermuteSpaceInPlace;
    tol=1e-4,
    compressor=ACA(; tol=tol),
    isnear=isnear(),
    nearmatrixdata=defaultmatrixdata(operator, testspace, trialspace),
    farmatrixdata=defaultfarmatrixdata(operator, testspace, trialspace),
    scheduler=DynamicScheduler(),
)
    testperm = permutation(testtree(tree))
    trialperm = permutation(trialtree(tree))

    permute!(testspace, testperm)
    !(testspace === trialspace) && permute!(trialspace, trialperm)

    nears = assemblenears_consecutive(
        operator,
        testspace,
        trialspace,
        tree;
        isnear=isnear,
        matrixdata=nearmatrixdata,
        scheduler=scheduler,
    )

    fars = assemblefars_consecutive(
        operator,
        testspace,
        trialspace,
        tree;
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
    nearquadstrat=defaultnearquadstrat(operator, space, space),
    farquadstrat=defaultfarquadstrat(operator, space, space),
    ntasks=Threads.nthreads(),
)
    return error("Symmetric version not implemented yet")
end
