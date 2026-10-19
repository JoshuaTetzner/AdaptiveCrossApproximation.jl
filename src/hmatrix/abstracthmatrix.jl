using LinearAlgebra
using LinearMaps
using OhMyThreads

"""
    defaultmatrixdata(operator, testspace, trialspace)

Return backend-specific data used to construct an [`AbstractKernelMatrix`](@ref)
during hierarchical assembly. The fallback returns `nothing`; extensions may return
quadrature rules or other reusable assembly data.
"""
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

"""
    defaultcompressor(operator, testspace, trialspace; tol=1e-4)

Return the compressor used by [`HMatrix`](@ref) and
[`AdaptiveCrossApproximation.assemble`](@ref) when none is supplied explicitly.
The fallback is `ACA(; tol)`. Extensions may select a more robust compressor for
particular operator and space combinations.
"""
defaultcompressor(operator, testspace, trialspace; tol::Real=1e-4) = ACA(; tol=tol)

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
_treepositions(space) = space

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

function buildtree(::AbstractTree, args...; kwargs...)
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

defaulttreebackend(testspace, trialspace) = defaulttreebackend()

# The ParallelKMeans tree extension builds BEAST trees. Point vectors use the
# H2Trees backend even when both extensions have been loaded.
function defaulttreebackend(::AbstractVector, ::AbstractVector)
    isnothing(Base.get_extension(AdaptiveCrossApproximation, :ACAH2Trees)) &&
        error("Please load H2Trees.jl to build a point-space cluster tree.")
    return H2Tree()
end

"""
    testtree(tree)
    trialtree(tree)
    BlockTree(testtree, trialtree)
    root(tree)
    children(tree, node)
    parent(tree, node)
    firstchild(tree, node)
    isleaf(tree, node)
    leaves(tree)
    center(tree, node)
    values(tree, node)
    numberofnodes(tree)
    numberofvalues(tree, node)
    levels(tree)
    LevelIterator(tree, level)

Tree backend interface required by [`HMatrix`](@ref) assembly and
[`TreeMimicryPivoting`](@ref). Names and semantics match H2Trees; the ACAH2Trees
extension forwards these methods to that backend. `testtree`/`trialtree` return
the row/column cluster trees, and `BlockTree` combines them. `values(tree, node)`
returns the indices in a node's subtree in traversal order; at the root these
give the tree-to-space permutation. `levels` lists levels coarsest first,
visited via `LevelIterator`.
"""
testtree(tree) = error("Requires implementation for $(typeof(tree))")
trialtree(tree) = error("Requires implementation for $(typeof(tree))")
BlockTree(testtree, trialtree) =
    error("Requires implementation for $(typeof(testtree)), $(typeof(trialtree))")
root(tree) = error("Requires implementation for $(typeof(tree))")
children(tree, node::Int) = error("Requires implementation for $(typeof(tree))")
parent(tree, node::Int) = error("Requires implementation for $(typeof(tree))")
firstchild(tree, node::Int) = error("Requires implementation for $(typeof(tree))")
isleaf(tree, node::Int) = error("Requires implementation for $(typeof(tree))")
leaves(tree) = error("Requires implementation for $(typeof(tree))")
center(tree, node::Int) = error("Requires implementation for $(typeof(tree))")
values(tree, node::Union{Int,Vector{Int}}) =
    error("Requires implementation for $(typeof(tree))")
numberofnodes(tree) = error("Requires implementation for $(typeof(tree))")
numberofvalues(tree, node::Int) = error("Requires implementation for $(typeof(tree))")
levels(tree) = error("Requires implementation for $(typeof(tree))")
LevelIterator(tree, level) = error("Requires implementation for $(typeof(tree))")

interactions(tree, isnear) =
    error("Interactions are not implemented for $(typeof(tree)), $(typeof(isnear))")
symmetricinteractions(tree, isnear) = error(
    "Symmetric interactions are not implemented for $(typeof(tree)), $(typeof(isnear))"
)

# ACA-specific mutation after physically permuting a space. H2Trees has no
# equivalent operation; only the leaf value vectors are modified, not topology.
function _settreeorder!(tree)
    start = 1
    for leaf in leaves(tree)
        leafvalues = values(tree, leaf)
        leafvalues .= start:(start + length(leafvalues) - 1)
        start += length(leafvalues)
    end
    return tree
end

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

The single-space call `ordering(tree, space)` uses only `testtree(tree)` and
permutes the space once. Both general and transpose-symmetric `HMatrix`
assembly support this ordering.
"""
struct PermuteSpaceInPlace <: SpaceOrderingStyle end
function (::PermuteSpaceInPlace)(tree, space)
    cluster = testtree(tree)
    permute!(space, collect(values(cluster, root(cluster))))
    _settreeorder!(cluster)
    return space
end

function (::PermuteSpaceInPlace)(tree, testspace, trialspace)
    rowtree = testtree(tree)
    coltree = trialtree(tree)
    if testspace === trialspace && rowtree === coltree
        # Case 1: same space and same tree → permute once
        permute!(testspace, collect(values(rowtree, root(rowtree))))
        _settreeorder!(rowtree)
        return testspace, testspace
    elseif testspace !== trialspace && rowtree !== coltree
        # Case 2: different spaces and different trees → permute each independently
        testperm = collect(values(rowtree, root(rowtree)))
        trialperm = collect(values(coltree, root(coltree)))
        permute!(testspace, testperm)
        permute!(trialspace, trialperm)
        _settreeorder!(rowtree)
        _settreeorder!(coltree)
        return testspace, trialspace
    elseif testspace === trialspace
        # Case 3: same space but different trees → copy trialspace before permuting
        @warn "testspace and trialspace are the same object but testtree ≠ trialtree. Copying trialspace and permuting independently."
        testperm = collect(values(rowtree, root(rowtree)))
        trialperm = collect(values(coltree, root(coltree)))
        trialspace_permuted = permute(trialspace, trialperm)
        permute!(testspace, testperm)
        _settreeorder!(rowtree)
        _settreeorder!(coltree)
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

The single-space call `ordering(tree, space)` returns the space unchanged.
"""
struct PreserveSpaceOrder <: SpaceOrderingStyle end
function (::PreserveSpaceOrder)(tree, space)
    return space
end

function (::PreserveSpaceOrder)(tree, testspace, trialspace)
    return testspace, trialspace
end

include("skeleton.jl")
include("blockstorage.jl")
include("hmatrix.jl")
include("treeindex.jl")
include("arithmetic/utils.jl")
include("arithmetic/matvec.jl")
include("arithmetic/blockscratch.jl")
include("arithmetic/taskgraph.jl")
include("arithmetic/recompression.jl")
include("arithmetic/workspace.jl")
include("arithmetic/blockview.jl")
include("arithmetic/hierarchicaladdition.jl")
include("arithmetic/thinmultiplication.jl")
include("arithmetic/hierarchicalmultiplication.jl")
include("arithmetic/leaflu.jl")
include("arithmetic/triangular.jl")
include("arithmetic/solveplan.jl")
include("arithmetic/lu.jl")
include("arithmetic/solve.jl")
include("arithmetic/operations.jl")
include("interactions.jl")
include("nearassembly.jl")
include("farassembly.jl")

"""
    HMatrix(operator, testspace, trialspace, tree; kwargs...)

Assemble a hierarchical matrix approximation of an operator on test and trial spaces.

# Arguments

  - `operator`: bilinear form or kernel operator used for matrix entry evaluation
  - `testspace`: test space used for row indexing
  - `trialspace`: trial space used for column indexing
  - `tree`: hierarchical clustering/tree structure controlling block partitioning
  - `tol`: compression tolerance (default `1e-4`). It is stored with the matrix
    and governs every later recompression of the hierarchical arithmetic.
  - `compressor`: ACA-style compressor. Defaults to
    `defaultcompressor(operator, testspace, trialspace; tol)`.
  - `isnear`: near-field predicate controlling admissibility
  - `maxrank`: maximum rank for far-field block compression
  - `spaceordering`: strategy for applying tree permutations to spaces
  - `nearmatrixdata`: optional data passed to near-field assembly
  - `farmatrixdata`: optional data passed to far-field assembly
  - `ntasks`: number of parallel assembly tasks (default `Threads.nthreads()`)
  - `verbose`: enable progress output for near- and far-field assembly (default `true`)

# Returns

An `HMatrix` containing assembled near and far interactions.

# Notes

Use this constructor as the main entry point when you already have a tree. For a convenience
entry point, use `assemble`.
This two-space constructor uses general assembly even when the spaces are the
same object. Use `HMatrix(operator, space, tree; kwargs...)` for a
transpose-symmetric operator.

# See also

`assemble`, `ACA`, `IACA`, `farmatrix`, `nearmatrix`
"""
function HMatrix(
    operator,
    testspace,
    trialspace,
    tree;
    tol=1e-4,
    compressor=defaultcompressor(operator, testspace, trialspace; tol=tol),
    isnear=isnear(),
    maxrank=40,
    spaceordering::SpaceOrderingStyle=PreserveSpaceOrder(),
    nearmatrixdata=defaultmatrixdata(operator, testspace, trialspace),
    farmatrixdata=defaultfarmatrixdata(operator, testspace, trialspace),
    ntasks::Integer=Threads.nthreads(),
    verbose::Bool=true,
)
    testspace, trialspace = spaceordering(tree, testspace, trialspace)
    partition = interactions(tree, isnear)
    near = assemblenears(
        operator,
        testspace,
        trialspace,
        tree,
        partition;
        matrixdata=nearmatrixdata,
        ntasks=ntasks,
        verbose=verbose,
    )
    far = assemblefars(
        operator,
        testspace,
        trialspace,
        tree,
        partition;
        compressor=compressor,
        maxrank=maxrank,
        matrixdata=farmatrixdata,
        ntasks=ntasks,
        verbose=verbose,
    )
    csr, csc = blockindices(tree, partition)
    dim = (length(testspace), length(trialspace))
    return HMatrix{eltype(eltype(near))}(tree, near, far, csr, csc, spaceordering, dim, tol)
end

"""
    HMatrix(operator, space, tree; kwargs...)

Assemble a transpose-symmetric H-matrix on one space. Only `testtree(tree)`
is used for both axes. The operator and any custom `isnear` predicate must
be transpose-symmetric; this assumption is not checked. Complex symmetric
operators use a transpose, without conjugation, for the opposite block.

Each terminal near pair is stored as one dense block. Far blocks use the
configured compressor once per unordered pair and share their factors with the
transposed logical block. Diagonal near blocks are assembled in full, then made
symmetric by copying their upper triangles into their lower triangles.

Compression, assembly-data and `ntasks` keywords follow the two-space
constructor. For automatic tree construction, use `assemble(operator, space)`.

# Examples

```julia
hmat = HMatrix(operator, space, tree; tol=1e-5)
```
"""
function HMatrix(
    operator,
    space,
    tree;
    tol=1e-4,
    compressor=defaultcompressor(operator, space, space; tol=tol),
    isnear=isnear(),
    maxrank=40,
    spaceordering::SpaceOrderingStyle=PreserveSpaceOrder(),
    nearmatrixdata=defaultmatrixdata(operator, space, space),
    farmatrixdata=defaultfarmatrixdata(operator, space, space),
    ntasks::Integer=Threads.nthreads(),
    verbose::Bool=true,
)
    cluster = testtree(tree)
    sharedtree = BlockTree(cluster, cluster)
    space = spaceordering(sharedtree, space)
    partition = symmetricinteractions(sharedtree, isnear)
    near = assemblenears(
        operator,
        space,
        sharedtree,
        partition;
        matrixdata=nearmatrixdata,
        ntasks=ntasks,
        verbose=verbose,
    )
    far = assemblefars(
        operator,
        space,
        space,
        sharedtree,
        partition;
        compressor=compressor,
        maxrank=maxrank,
        matrixdata=farmatrixdata,
        ntasks=ntasks,
        verbose=verbose,
    )
    csr, csc = symmetricblockindices(sharedtree, partition)
    dim = (length(space), length(space))
    return HMatrix{eltype(eltype(near))}(
        sharedtree, near, far, csr, csc, spaceordering, dim, tol
    )
end
