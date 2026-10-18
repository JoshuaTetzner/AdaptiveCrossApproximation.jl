"""
    IsNearFunctor{F,CrossLevel}

Admissibility predicate with cross-level traversal encoded as a Boolean type
parameter. `η` is either the geometric separation parameter or a custom callable
predicate. Symmetry is selected by the interaction traversal function.
"""
struct IsNearFunctor{F,CrossLevel}
    η::F
end

function IsNearFunctor(η; crosslevel::Bool=false)
    return IsNearFunctor{typeof(η),crosslevel}(η)
end

function IsNearFunctor(
    predicate::IsNearFunctor{F,CrossLevel}; crosslevel::Bool=CrossLevel
) where {F,CrossLevel}
    return IsNearFunctor{F,crosslevel}(predicate.η)
end

function (predicate::IsNearFunctor)(args...)
    return predicate.η(args...)
end

"""
    isnear(η::Real=1.0; crosslevel=false)
    isnear(predicate; crosslevel=false)

Create an admissibility predicate for near-field block detection. The predicate
returns `true` for cluster pairs requiring dense assembly or subdivision, and
`false` for admissible far-field pairs. For the H2Trees backend, larger `η`
classifies more cluster pairs as far field. A custom callable may replace `η`.

Cross-level traversal is reserved for future implementation and currently requires
`crosslevel=false`. When given an existing `IsNearFunctor`, its cross-level setting
is preserved unless explicitly overridden.
"""
function isnear(criterion=1.0; kwargs...)
    return IsNearFunctor(criterion; kwargs...)
end

_checkcrosslevel(::IsNearFunctor{F,false}) where {F} = nothing
function _checkcrosslevel(::IsNearFunctor{F,true}) where {F}
    return throw(ArgumentError("Cross-level interaction traversal is not implemented yet"))
end

"""
    nearinteractions(tree; isnear=isnear(), spaceordering=PreserveSpaceOrder())
    nearinteractions(tree, isnear, spaceordering)

Extract near-field cluster pairs. The keyword entry point forwards to positional
methods implemented by tree backends, dispatching on the admissibility functor
and space ordering.

Return `(values, nearvalues)`, with row indices and corresponding near-field
column indices. Use `symmetricnearinteractions` to visit each unordered pair once.

`PreserveSpaceOrder()` returns indices in the original space order.
`PermuteSpaceInPlace()` returns indices for spaces already permuted to tree order.
The traversal does not permute spaces itself.
"""
function nearinteractions(
    tree; isnear=isnear(), spaceordering::SpaceOrderingStyle=PreserveSpaceOrder(), kwargs...
)
    return nearinteractions(tree, IsNearFunctor(isnear), spaceordering; kwargs...)
end

function nearinteractions(
    tree, isnear::IsNearFunctor, spaceordering::SpaceOrderingStyle; kwargs...
)
    return error(
        "Near interactions are not implemented for $(typeof(tree)), " *
        "$(typeof(isnear)), $(typeof(spaceordering))",
    )
end

"""
    symmetricnearinteractions(tree; isnear=isnear(), spaceordering=PreserveSpaceOrder())
    symmetricnearinteractions(tree, isnear, spaceordering)

Return `(diagindices, rowindices, colindices)` for self clusters and unordered
off-diagonal near interactions. Each row cluster occurs once in `rowindices`,
with its off-diagonal near columns combined in `colindices`, in tree-node order
as in general assembly. Only the test cluster tree is used for both axes.
The admissibility predicate must be transpose-symmetric.

`diagindices[i]` describes a square self block. Each `(rowindices[i], colindices[i])`
describes one rectangular off-diagonal block; its transpose supplies the opposite
orientation.

`PreserveSpaceOrder()` returns indices in the original space order.
`PermuteSpaceInPlace()` returns indices for spaces already permuted to tree order.
The keyword entry point forwards to positional backend methods for each ordering.
Both orderings support traversal, although symmetric matrix assembly currently
supports only `PreserveSpaceOrder()`.
"""
function symmetricnearinteractions(
    tree; isnear=isnear(), spaceordering::SpaceOrderingStyle=PreserveSpaceOrder(), kwargs...
)
    return symmetricnearinteractions(tree, IsNearFunctor(isnear), spaceordering; kwargs...)
end

function symmetricnearinteractions(
    tree, isnear::IsNearFunctor, spaceordering::SpaceOrderingStyle; kwargs...
)
    return error(
        "Symmetric near interactions are not implemented for $(typeof(tree)), " *
        "$(typeof(isnear)), $(typeof(spaceordering))",
    )
end

"""
    assemblenears(operator, testspace, trialspace, tree, ::PreserveSpaceOrder; kwargs...)

Assemble near-field blocks without reordering test and trial spaces.

Computes dense (non-low-rank) matrix blocks for all near-field cluster pairs,
retaining the original ordering of test and trial spaces. The resulting
block-sparse matrix stores these near-field interactions.

# Arguments

  - `operator`: Operator/kernel for matrix entry evaluation
  - `testspace`: Test space (row basis/evaluation points)
  - `trialspace`: Trial space (column basis/evaluation points)
  - `tree`: Hierarchical tree structure
  - `::PreserveSpaceOrder`: Space ordering strategy marker
  - `isnear`: Admissibility predicate (default: `isnear()`)
  - `ntasks`: number of parallel tasks for the assembly and the products of the
    result (default `Threads.nthreads()`). With `ntasks=1`, no near-field
    conflict graph is built.
  - `matrixdata`: Assembly data passed to kernel matrix (optional)
  - `verbose`: enable progress output (default `true`)

# Returns

  - `BlockSparseMatrix`: Block-sparse storage of near-field blocks

# Notes

Called by `HMatrix` constructor when `spaceordering=PreserveSpaceOrder()`. Each
near-field block is evaluated directly via the kernel matrix, avoiding compression.
"""
function assemblenears(
    operator,
    testspace,
    trialspace,
    tree,
    spaceordering::PreserveSpaceOrder;
    isnear=isnear(),
    ntasks::Integer=Threads.nthreads(),
    matrixdata=defaultmatrixdata(operator, testspace, trialspace),
    verbose::Bool=true,
)
    nearmatrix = AbstractKernelMatrix(
        operator, testspace, trialspace; matrixdata=matrixdata
    )
    values, nearvalues = nearinteractions(tree; spaceordering=spaceordering, isnear=isnear)

    isempty(values) && return BlockSparseMatrix(
        Matrix{eltype(nearmatrix)}[], Vector{Int}[], Vector{Int}[], size(nearmatrix)
    )

    blocks = [
        zeros(eltype(nearmatrix), length(values[i]), length(nearvalues[i])) for
        i in eachindex(values)
    ]
    pbar = Progress(length(blocks); desc="Assembling near interactions: ", enabled=verbose)
    assemble_blocks(nearmatrix, blocks, values, nearvalues; ntasks=ntasks, pbar=pbar)
    finish!(pbar)

    nears = BlockSparseMatrix(
        blocks, values, nearvalues, size(nearmatrix); scheduler=_scheduler(ntasks)
    )

    return nears
end

function splitblock(block::Matrix{T}, lens::Vector{Int}) where {T}
    return [
        view(block, 1:size(block, 1), sum(lens[1:(i - 1)]) .+ (1:lens[i])) for
        i in eachindex(lens)
    ]
end

"""
    assemblenears(operator, testspace, trialspace, tree, ::PermuteSpaceInPlace; kwargs...)

Assemble near-field blocks with tree-aligned space reordering.

Computes dense matrix blocks for near-field cluster pairs, with both spaces
permuted to align with the hierarchical tree structure. Produces a specialized
block storage format optimized for the reordered layout.

# Arguments

  - `operator`: Operator/kernel for matrix entry evaluation
  - `testspace`: Test space (row basis) — will be reordered
  - `trialspace`: Trial space (column basis) — will be reordered
  - `tree`: Hierarchical tree structure
  - `::PermuteSpaceInPlace`: Space ordering strategy marker
  - `isnear`: Admissibility predicate (default: `isnear()`)
  - `ntasks`: number of parallel tasks (default `Threads.nthreads()`)
  - `matrixdata`: Assembly data passed to kernel matrix (optional)
  - `verbose`: enable progress output (default `true`)

# Returns

  - `VariableBlockCompressedRowStorage`: Permuted block-sparse near-field storage

# Notes

Called by `HMatrix` constructor when `spaceordering=PermuteSpaceInPlace()`. The
spaces are reordered to match tree leaf ordering before block assembly, improving
cache locality and reducing block fragmentation.
"""
function assemblenears(
    operator,
    testspace,
    trialspace,
    tree,
    spaceordering::PermuteSpaceInPlace;
    isnear=isnear(),
    ntasks::Integer=Threads.nthreads(),
    matrixdata=defaultmatrixdata(operator, testspace, trialspace),
    verbose::Bool=true,
)
    nearmatrix = AbstractKernelMatrix(
        operator, testspace, trialspace; matrixdata=matrixdata
    )
    values, nearvalues = nearinteractions(tree; spaceordering=spaceordering, isnear=isnear)
    blocks = zeros.(
        eltype(nearmatrix), length.(values), [sum(length.(n)) for n in nearvalues]
    )
    # There should be a prettier not hardcoded way to do this, but it works for now
    viewblocks = Vector{
        Vector{
            SubArray{
                eltype(nearmatrix),
                2,
                Matrix{eltype(nearmatrix)},
                Tuple{UnitRange{Int},UnitRange{Int}},
                false,
            },
        },
    }(
        undef, length(blocks)
    )
    pbar = Progress(length(blocks); desc="Assembling near interactions: ", enabled=verbose)
    scheduler = _scheduler(ntasks)
    @tasks for i in eachindex(blocks)
        @set scheduler = scheduler
        nearmatrix(blocks[i], values[i], Iterators.flatten(nearvalues[i]))
        viewblocks[i] = splitblock(blocks[i], length.(nearvalues[i]))
        next!(pbar)
    end
    finish!(pbar)
    mat = VariableBlockCompressedRowStorage{
        eltype(nearmatrix),eltype(Iterators.flatten(viewblocks)),Int,typeof(scheduler)
    }(
        collect(Iterators.flatten(viewblocks)),
        [1; cumsum(length.(nearvalues)) .+ 1],
        first.(Iterators.flatten(nearvalues)),
        first.(values),
        size(nearmatrix),
        scheduler,
    )
    return mat
end

"""
    assemblenears(operator, space, tree, spaceordering; kwargs...)

Assemble a transpose-symmetric near field. One final-sized off-diagonal block
is assembled per row cluster, with all its near columns combined by the
interaction traversal. Diagonal blocks are assembled separately in full, then
their upper triangles are mirrored into their lower triangles.

Only `PreserveSpaceOrder()` is supported; symmetric VBCRS storage is required
before `PermuteSpaceInPlace()` can be used.
"""
function assemblenears(
    operator,
    space,
    tree,
    spaceordering::SpaceOrderingStyle;
    isnear=isnear(),
    ntasks::Integer=Threads.nthreads(),
    matrixdata=defaultmatrixdata(operator, space, space),
    verbose::Bool=true,
)
    _check_symmetric_ordering(spaceordering)
    scheduler = _scheduler(ntasks)
    nearmatrix = AbstractKernelMatrix(operator, space, space; matrixdata=matrixdata)
    diagindices, rowindices, colindices = symmetricnearinteractions(
        tree; spaceordering=spaceordering, isnear=isnear
    )
    T = eltype(nearmatrix)
    # Typed comprehensions: Julia 1.10 infers Vector{Any} for an empty block list.
    diagonals = Matrix{T}[
        zeros(T, length(indices), length(indices)) for indices in diagindices
    ]
    offdiagonals = Matrix{T}[
        zeros(T, length(rowindices[i]), length(colindices[i])) for
        i in eachindex(rowindices)
    ]

    pbar = Progress(
        length(offdiagonals) + length(diagonals);
        desc="Assembling symmetric near interactions: ",
        enabled=verbose,
    )
    assemble_blocks(
        nearmatrix, offdiagonals, rowindices, colindices; ntasks=ntasks, pbar=pbar
    )
    assemble_blocks(
        nearmatrix, diagonals, diagindices, diagindices; ntasks=ntasks, pbar=pbar
    )
    finish!(pbar)
    for block in diagonals
        for j in axes(block, 2), i in (j + 1):size(block, 1)
            block[i, j] = block[j, i]
        end
    end

    # BlockSparseMatrices 0.3.1 does not construct SymmetricBlockMatrix with no
    # off-diagonal blocks. A diagonal-only near field needs no mirrored MVP.
    if isempty(diagonals) && isempty(offdiagonals)
        return BlockSparseMatrix(
            diagonals,
            diagindices,
            diagindices,
            size(nearmatrix);
            scheduler=SerialScheduler(),
        )
    end
    if isempty(offdiagonals)
        return BlockSparseMatrix(
            diagonals, diagindices, diagindices, size(nearmatrix); scheduler=scheduler
        )
    end
    if scheduler isa SerialScheduler
        return _serialsymmetricblockmatrix(
            diagonals,
            diagindices,
            offdiagonals,
            rowindices,
            colindices,
            size(nearmatrix),
            scheduler,
        )
    end
    # The 0.3.1 parallel constructor needs a nonempty diagonal block list to
    # build its conflict graph. A zero block supplies one when needed.
    if isempty(diagonals)
        push!(diagonals, zeros(T, 1, 1))
        push!(diagindices, [1])
    end
    return SymmetricBlockMatrix(
        diagonals,
        diagindices,
        offdiagonals,
        rowindices,
        colindices,
        size(nearmatrix);
        scheduler=scheduler,
    )
end
