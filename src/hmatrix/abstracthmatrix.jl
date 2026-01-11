module H

using AdaptiveCrossApproximation
using BlockSparseMatrices
using OhMyThreads
using LinearMaps
using LinearAlgebra

defaultnearquadstrat(operator, testspace, trialspace) = error("Not implemented")
defaultfarquadstrat(operator, testspace, trialspace) = error("Not implemented")

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
    permutation=true,
    nearquadstrat=defaultnearquadstrat(operator, testspace, trialspace),
    farquadstrat=defaultfarquadstrat(operator, testspace, trialspace),
    ntasks=Threads.nthreads(),
)
    if permutation
        permute!(testspace, testtree(tree))
        permute!(trialspace, trialtree(tree))
    else
        (tpermutation, testspace=permute(testspace, testtree(tree)))
        (spermutation, trialspace=permute(trialspace, trialtree(tree)))
    end

    nears = assemblenears(
        operator,
        testspace,
        trialspace,
        tree;
        isnear=isnear,
        ntasks=ntasks,
        quadstrat=nearquadstrat,
    )

    fars = assemblefars(
        operator,
        testspace,
        trialspace,
        tree;
        compressor=compressor,
        isnear=isnear,
        quadstrat=farquadstrat,
        ntasks=ntasks,
    )

    permutation && return HMatrix{scalartype(operator)}(
        nears, fars, (length(testspace), length(trialspace)), ntasks
    )
    return PermutedHMatrix(
        (tpermutation, spermutation),
        HMatrix{scalartype(operator)}(
            nears, fars, (length(testspace), length(trialspace)), ntasks
        ),
    )
end
#=
function HMatrix(
    operator,
    space,
    tree;
    isnear=IsNearFunctor(),
    compressor=ACA(; tol=1e-4),
    permutation=true,
    nearquadstrat=defaultnearquadstrat(operator, space, space),
    farquadstrat=defaultfarquadstrat(operator, space, space),
    ntasks=Threads.nthreads(),
)
    if permutation
        permute!(space, testtree(tree))
    else
        (permutation, space=permute(space, testtree(tree)))
    end

    nears = assemblenears(
        operator, space, tree; isnear=isnear, ntasks=ntasks, quadstrat=nearquadstrat
    )

    fars = assemblefars(
        operator,
        space,
        tree;
        isnear=isnear,
        quadstrat=farquadstrat,
        compressor=compressor,
        ntasks=ntasks,
    )

    permutation && return HMatrix{scalartype(operator)}(
        nears, fars, (length(space), length(space)), ntasks
    )
    return PermutedHMatrix(
        (permutation, permutation),
        HMatrix{scalartype(operator)}(nears, fars, (length(space), length(space)), ntasks),
    )
end
=#
end
