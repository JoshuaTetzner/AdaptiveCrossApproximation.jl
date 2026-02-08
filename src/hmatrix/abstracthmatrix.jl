using BlockSparseMatrices
using LinearAlgebra
using LinearMaps
using OhMyThreads

defaultnearquadstrat(operator, testspace, trialspace) = error("Not implemented")
defaultfarquadstrat(operator, testspace, trialspace) = error("Not implemented")
scalartype(operator) = error("Not implemented for $(typeof(operator))")

testtree(tree) = error("Requiers implementation for $(typeof(tree))")
trialtree(tree) = error("Requiers implementation for $(typeof(tree))")
levels(tree) = error("Requiers implementationfor $(typeof(tree))")
LevelIterator(tree, level) = error("Requiers implementation for $(typeof(tree))")

permutation(tree) = error("Requiers implementation for $(typeof(tree))")
permute(space, perm) =
    error("Requiers implementation for $(typeof(space)) and $(typeof(perm))")
permute!(space, perm) =
    error("Requiers implementation for $(typeof(space)) and $(typeof(perm))")

include("kernelmatrix/abstractkernelmatrix.jl")
include("kernelmatrix/beastkernelmatrix.jl")
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
    isnear=isnear(),
    compressor=ACA(; tol=1e-4),
    perm=true,
    nearquadstrat=defaultnearquadstrat(operator, testspace, trialspace),
    farquadstrat=defaultfarquadstrat(operator, testspace, trialspace),
    scheduler=DynamicScheduler(),
)
    testperm = permutation(testtree(tree))
    trialperm = permutation(trialtree(tree))
    if perm
        permute!(testspace, testperm)
        permute!(trialspace, trialperm)
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
        quadstrat=nearquadstrat,
        scheduler=scheduler,
    )

    fars = assemblefars(
        operator,
        testspace,
        trialspace,
        tree;
        compressor=compressor,
        isnear=isnear,
        quadstrat=farquadstrat,
        scheduler=scheduler,
    )

    perm && return HMatrix{scalartype(operator)}(
        nears, fars, (length(testspace), length(trialspace))
    )
    return PermutedHMatrix(
        (testperm, trialperm),
        HMatrix{scalartype(operator)}(nears, fars, (length(testspace), length(trialspace))),
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
