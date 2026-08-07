using BlockSparseMatrices
using LinearAlgebra
using LinearMaps
using OhMyThreads

defaultmatrixdata(operator, testspace, trialspace) = nothing

"""
    tofarquadstrat(quadstrat)

Convert a near-field quadrature strategy into the strategy used for admissible
(far-field, well-separated) blocks.

Falls back to returning `quadstrat` unchanged. The `ACABEAST` extension specializes
this for `BEAST.DoubleNumWiltonSauterQStrat` to extract its non-singular
`outer_rule_far`/`inner_rule_far` orders into a plain `BEAST.DoubleNumQStrat`, since
far blocks are by construction well-separated and never need singular quadrature.
"""
tofarquadstrat(quadstrat) = quadstrat

defaultfarmatrixdata(operator, testspace, trialspace) =
    tofarquadstrat(defaultmatrixdata(operator, testspace, trialspace))
defaultcompressor(operator, testspace, trialspace) = ACA(; tol=1e-4)

# kernelmatrix code
"""
    scalartype(operator)

Return the scalar element type produced by evaluating `operator`.

Interface point for custom operators used with [`AbstractKernelMatrix`](@ref)
backends; e.g. the `ACABEAST` extension implements this for BEAST integral
operators. No default is provided — implement a method for your operator type.
"""
scalartype(operator) = error("Not implemented for $(typeof(operator))")
permute(space, perm) = permute!(copy(space), perm)

# tree code
abstract type AbstractTree end

"""
    H2Tree <: AbstractTree

Tree backend marker selecting `H2Trees.jl`'s `TwoNTree` clustering, built by the
`ACAH2Trees` extension.
"""
struct H2Tree <: AbstractTree end

"""
    KMeansTreeBackend <: AbstractTree

Tree backend marker selecting k-means clustering (via `ParallelKMeans.jl`), built
by the `ACAParallelKMeansTrees` extension.
"""
struct KMeansTreeBackend <: AbstractTree end

function _tree(::AbstractTree, args...; kwargs...)
    return error("Please load H2Trees.jl or your custom tree implementation.")
end

"""
    defaulttreebackend()

Return the tree backend `assemble` uses when no `tree` is given explicitly.

Resolved at call time from which packages are loaded: [`KMeansTreeBackend`](@ref)
if `ParallelKMeans.jl` is loaded alongside `H2Trees.jl` (via the `ACAParallelKMeansTrees`
extension), otherwise [`H2Tree`](@ref) if `H2Trees.jl` alone is loaded (via `ACAH2Trees`).
Errors if neither is loaded, since `assemble` then has no way to build a tree.
"""
function defaulttreebackend()
    if !isnothing(Base.get_extension(AdaptiveCrossApproximation, :ACAParallelKMeansTrees))
        return KMeansTreeBackend()
    elseif !isnothing(Base.get_extension(AdaptiveCrossApproximation, :ACAH2Trees))
        return H2Tree()
    else
        return error(
            "assemble needs a clustering tree backend: load H2Trees.jl (optionally " *
            "with ParallelKMeans.jl for k-means clustering), or pass `tree=` explicitly.",
        )
    end
end

"""
    testtree(tree)
    trialtree(tree)
    levels(tree)
    LevelIterator(tree, level)
    permutation(tree)

Tree backend interface required by [`HMatrix`](@ref) assembly and
[`TreeMimicryPivoting`](@ref); implement these for a custom tree type (the
ACAH2Trees extension implements them for `H2Trees.TwoNTree`). `testtree`/`trialtree`
return the row/column clustering sub-tree (for a single shared tree, this may
return `tree` itself); `levels` returns tree levels coarsest first, visited via
`LevelIterator`; `permutation` returns the index permutation the tree applies to
its underlying space, used by [`PermuteSpaceInPlace`](@ref).
"""
testtree(tree) = error("Requires implementation for $(typeof(tree))")
trialtree(tree) = error("Requires implementation for $(typeof(tree))")
levels(tree) = error("Requires implementation for $(typeof(tree))")
LevelIterator(tree, level) = error("Requires implementation for $(typeof(tree))")
permutation(tree) = error("Requires implementation for $(typeof(tree))")

"""
    SpaceOrderingStyle

Abstract type selecting how test/trial spaces are reconciled with a tree's
internal point ordering during [`HMatrix`](@ref) assembly.

# Concrete Types

  - [`PreserveSpaceOrder`](@ref): Leave spaces untouched, index through the tree's
    permutation instead (the default).
  - [`PermuteSpaceInPlace`](@ref): Physically permute spaces to match tree order.
"""
abstract type SpaceOrderingStyle end

"""
    PermuteSpaceInPlace <: SpaceOrderingStyle

`spaceordering` style that permutes `testspace`/`trialspace` in place to match the
tree's internal point ordering, rather than leaving them untouched.

Pass as `HMatrix(...; spaceordering=PermuteSpaceInPlace())`. Useful when downstream
code expects a space physically reordered to tree order; compare with the default
[`PreserveSpaceOrder`](@ref).
"""
struct PermuteSpaceInPlace <: SpaceOrderingStyle end
function (::PermuteSpaceInPlace)(tree, testspace, trialspace)
    testperm = permutation(testtree(tree))
    if testspace === trialspace && testtree(tree) === trialtree(tree)
        # Case 1: same space and same tree → permute once
        permute!(testspace, testperm)
        return testspace, testspace
    elseif testspace !== trialspace && testtree(tree) !== trialtree(tree)
        # Case 2: different spaces and different trees → permute each independently
        permute!(testspace, testperm)
        permute!(trialspace, permutation(trialtree(tree)))
        return testspace, trialspace
    elseif testspace === trialspace
        # Case 3: same space but different trees → copy trialspace before permuting
        @warn "testspace and trialspace are the same object but testtree ≠ trialtree. Copying trialspace and permuting independently."
        trialspace_permuted = permute(trialspace, permutation(trialtree(tree)))
        permute!(testspace, testperm)
        return testspace, trialspace_permuted
    else
        error(
            "testspace ≠ trialspace but testtree === trialtree: permutation is ambiguous."
        )
    end
end
"""
    PreserveSpaceOrder <: SpaceOrderingStyle

`spaceordering` style that leaves `testspace`/`trialspace` untouched; block
assembly indexes through the tree's permutation instead. This is the default
for `HMatrix(...; spaceordering=PreserveSpaceOrder())`. Compare with
[`PermuteSpaceInPlace`](@ref).
"""
struct PreserveSpaceOrder <: SpaceOrderingStyle end
function (::PreserveSpaceOrder)(tree, testspace, trialspace)
    return testspace, trialspace
end

include("hmatrix.jl")
include("nearinteractions.jl")
include("skeleton.jl")
include("farinteractions.jl")

"""
    HMatrix(operator, testspace, trialspace, tree; kwargs...)

Assemble a hierarchical matrix approximation of an operator on test and trial spaces.

# Arguments

  - `operator`: bilinear form or kernel operator used for matrix entry evaluation
  - `testspace`: test space used for row indexing
  - `trialspace`: trial space used for column indexing
  - `tree`: hierarchical clustering/tree structure controlling block partitioning
  - `tol`: compression tolerance (default `1e-4`)
  - `compressor`: ACA-style compressor, e.g. `ACA(; tol=tol)`
  - `isnear`: near-field predicate controlling admissibility
  - `maxrank`: maximum rank for far-field block compression
  - `spaceordering`: strategy for applying tree permutations to spaces
  - `nearmatrixdata`: optional data passed to near-field assembly
  - `farmatrixdata`: optional data passed to far-field assembly
  - `scheduler`: thread scheduler used for assembly
  - `verbose`: enable progress output for near- and far-field assembly (default `true`)

# Returns

An `HMatrix` containing assembled near and far interactions.

# Notes

Use this constructor as the main entry point when you already have a tree. For a convenience
entry point, use `assemble`.

# See also

`assemble`, `ACA`, `IACA`, `farmatrix`, `nearmatrix`
"""
function HMatrix(
    operator,
    testspace,
    trialspace,
    tree;
    tol=1e-4,
    compressor=ACA(; tol=tol),
    isnear=isnear(),
    maxrank=40,
    spaceordering::SpaceOrderingStyle=PreserveSpaceOrder(),
    nearmatrixdata=defaultmatrixdata(operator, testspace, trialspace),
    farmatrixdata=defaultfarmatrixdata(operator, testspace, trialspace),
    scheduler=DynamicScheduler(),
    verbose::Bool=true,
)
    testspace, trialspace = spaceordering(tree, testspace, trialspace)

    nears = assemblenears(
        operator,
        testspace,
        trialspace,
        tree,
        spaceordering;
        isnear=isnear,
        matrixdata=nearmatrixdata,
        scheduler=scheduler,
        verbose=verbose,
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
        verbose=verbose,
    )

    return HMatrix{eltype(nears)}(nears, fars, (length(testspace), length(trialspace)))
end

function HMatrix(
    operator,
    space,
    tree;
    tol=1e-4,
    compressor=ACA(; tol=tol),
    isnear=isnear(),
    maxrank=40,
    spaceordering::SpaceOrderingStyle=PreserveSpaceOrder(),
    nearmatrixdata=defaultmatrixdata(operator, space, space),
    farmatrixdata=defaultfarmatrixdata(operator, space, space),
    scheduler=DynamicScheduler(),
    verbose::Bool=true,
)
    return error("Symmetric version not implemented yet")
end

"""
    farmatrix(hmat::HMatrix)

Extract the far-field (low-rank compressed) contributions from a hierarchical matrix.

Returns a new `HMatrix` containing only the low-rank far-field blocks, with all
near-field blocks removed. Useful for analyzing the rank structure or applying
operations that exploit low-rank properties.

# Arguments

  - `hmat::HMatrix`: Source hierarchical matrix

# Returns

  - `HMatrix`: New matrix with identical far-field interactions but empty near-field
"""
function farmatrix(hmat::HMatrix)
    blocks = Matrix{eltype(hmat)}[]
    nears = BlockSparseMatrix(blocks, Vector{Int}[], Vector{Int}[], hmat.dim)

    return HMatrix{eltype(hmat)}(nears, hmat.farinteractions, hmat.dim)
end

"""
    nearmatrix(hmat::HMatrix)

Extract the near-field (dense block) contributions from a hierarchical matrix.

Returns only the block-sparse near-field interactions, removing all low-rank
far-field blocks. Useful for analyzing near-field structure or applying
operations specific to dense blocks.

# Arguments

  - `hmat::HMatrix`: Source hierarchical matrix

# Returns

  - `BlockSparseMatrix`: Near-field interactions as block-sparse matrix
"""
function nearmatrix(hmat::HMatrix)
    return hmat.nearinteractions
end

"""
    storage(hmat::HMatrix)

Analyze and report memory storage requirements of a hierarchical matrix.

Prints detailed statistics (to stdout) comparing the actual storage needed
for the hierarchical representation versus dense storage, including compression
ratio. Also reports total memory footprint via Julia's `summarysize`.

# Arguments

  - `hmat::HMatrix`: Hierarchical matrix to analyze

# Returns

  - `Float64`: Total storage in GB used by hierarchical blocks

# Output

Prints to stdout:

  - `storage`: Total block storage in GB
  - `summary size`: Total memory footprint including Julia object overhead (GB)
  - `compression ratio`: Ratio of hierarchical storage to dense storage
"""
function storage(hmat::HMatrix)
    refsize = size(hmat, 1) * size(hmat, 2) * sizeof(eltype(hmat))
    matsize = 0
    for blk in hmat.nearinteractions.blocks
        matsize += length(blk)
    end
    for farmat in hmat.farinteractions
        for blk in farmat.blocks
            matsize += length(blk.U) + length(blk.V)
        end
    end
    println("storage: ", matsize * sizeof(eltype(hmat)) * 10^-9, " GB")
    println("summary size: ", Base.summarysize(hmat) * 10^-9, " GB")
    println("compression ratio: ", (matsize * sizeof(eltype(hmat))) / refsize)
    return matsize * sizeof(eltype(hmat)) * 10^-9
end
