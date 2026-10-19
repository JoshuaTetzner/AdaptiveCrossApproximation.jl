"""
    IsNearFunctor{F,CrossLevel}

Geometric admissibility functor with separation parameter `η` and cross-level
traversal encoded as a Boolean type parameter. Symmetry is selected by the
interaction traversal function.
"""
struct IsNearFunctor{F,CrossLevel}
    η::F
end

"""
    isnear(η::Real=1.0; crosslevel=false)

Create a geometric admissibility functor for near-field block detection. The functor
returns `true` for cluster pairs requiring dense assembly or subdivision, and
`false` for admissible far-field pairs. For the H2Trees backend, larger `η`
classifies more cluster pairs as far field. Custom callables can instead be passed
directly as the `isnear` argument of assembly and interaction traversal functions.

Cross-level traversal is reserved for future implementation and currently requires
`crosslevel=false`.
"""
function isnear(η::Real=1.0; crosslevel::Bool=false)
    return IsNearFunctor{typeof(η),crosslevel}(η)
end

_checkcrosslevel(isnear) = nothing
function _checkcrosslevel(::IsNearFunctor{F,true}) where {F}
    return throw(ArgumentError("Cross-level interaction traversal is not implemented yet"))
end

"""
    interactions(tree; isnear=isnear())
    interactions(tree, isnear)

Terminal blocks of the block tree `tree`: the admissible (far-field) cluster pairs
and the near-field pairs of which at least one cluster is a leaf. Tree backends
implement the positional method.

Return a named tuple `(; nearptr, nearnodes, farptr, farnodes)` in compressed row
storage over all nodes of `testtree(tree)`: the near blocks of test node `τ` are the
trial nodes `nearnodes[nearptr[τ]:(nearptr[τ + 1] - 1)]`, its far blocks
`farnodes[farptr[τ]:(farptr[τ + 1] - 1)]`. The position of a block in `nearnodes` or
`farnodes` is its position in the near or far storage of an [`HMatrix`](@ref). The
index sets of a node are given by `values(tree, node)`. Use
[`symmetricinteractions`](@ref) to visit each unordered pair once.
"""
function interactions(tree; isnear=isnear())
    return interactions(tree, isnear)
end

"""
    symmetricinteractions(tree; isnear=isnear())
    symmetricinteractions(tree, isnear)

Terminal blocks of a transpose-symmetric block tree, each unordered pair once. Only
the test cluster tree is used for both axes; the admissibility predicate must be
transpose-symmetric.

Return the format of [`interactions`](@ref), restricted to pairs `(a, b)` with
`a <= b`. The opposite block of `a < b` is the transpose of `(a, b)`.
"""
function symmetricinteractions(tree; isnear=isnear())
    return symmetricinteractions(tree, isnear)
end

# The terminal blocks from the per-node lists of near and far trial nodes.
function _interactions(nearnodes::Vector{Vector{Int}}, farnodes::Vector{Vector{Int}})
    nearptr, near = linearizestorage(nearnodes)
    farptr, far = linearizestorage(farnodes)
    return (; nearptr, nearnodes=near, farptr, farnodes=far)
end

# Test node of every entry of the compressed row storage `ptr`.
_rownodes(ptr) = [τ for τ in 1:(length(ptr) - 1) for _ in ptr[τ]:(ptr[τ + 1] - 1)]
