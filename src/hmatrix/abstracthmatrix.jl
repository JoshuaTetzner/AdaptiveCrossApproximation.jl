using BlockSparseMatrices
using LinearAlgebra
using LinearMaps
using OhMyThreads

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
    tol=1e-4,
    compressor=ACA(; tol=tol),
    isnear=isnear(),
    perm=true,
    nearmatrixdata=defaultmatrixdata(operator, testspace, trialspace),
    farmatrixdata=defaultfarmatrixdata(operator, testspace, trialspace),
    scheduler=DynamicScheduler(),
)
    testperm = permutation(testtree(tree))
    trialperm = permutation(trialtree(tree))
    if perm
        permute!(testspace, testperm)
        !(testspace === trialspace) && permute!(trialspace, trialperm)
    else
        testspace = permute(testspace, testperm)
        trialspace = permute(trialspace, trialperm)
    end

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

    perm &&
        return HMatrix{eltype(nears)}(nears, fars, (length(testspace), length(trialspace)))
    return PermutedHMatrix(
        (testperm, trialperm),
        HMatrix{eltype(nears)}(nears, fars, (length(testspace), length(trialspace))),
    )
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
