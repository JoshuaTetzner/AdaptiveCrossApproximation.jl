struct IsNearFunctor{F}
    η::F
end

"""
    isnear(η::Real=Float64(1.0))

Create an admissibility predicate for near-field block detection.

Clusters with scaled geometric distance less than `η` are considered admissible
for near-field assembly (direct evaluation). The scaling factor η controls the
geometric separation required for far-field low-rank approximation.

# Arguments

  - `η::Real = 1.0`: Admissibility parameter (dimensionless geometric distance threshold)

# Returns

  - `IsNearFunctor`: Admissibility predicate functor

# Notes

For well-separated clusters (distance > η × cluster diameter), the interaction
is computed via ACA compression; otherwise, via direct near-field assembly.
Typical values: `η ≈ 1.0` to `3.0` depending on required accuracy.
"""
function isnear(η::Real=1.0)
    return IsNearFunctor{typeof(η)}(η)
end

"""
    nearinteractions(tree; args...)

Extract near-field (admissible) block pairs from a hierarchical tree structure.

Should be implemented by tree backends (e.g., H2Trees) to return index ranges
for pairs of clusters that satisfy the admissibility criterion. This is the
primary method when space ordering is not modified.

# Arguments

  - `tree`: Hierarchical tree object implementing the tree interface
  - `args...`: Additional keyword arguments passed by the assembly routine (e.g., `isnear`)

# Returns

  - `(values, nearvalues)`: Tuple of index vectors for row/column near-field blocks

# Notes

Implemented by tree backend extensions (e.g., ACAH2Trees for H2Trees). Should
return rows and corresponding near-field column indices that are admissible for
direct (non-low-rank) assembly.
"""
function nearinteractions(tree; args...)
    return error("Needs to be implemented for $(typeof(tree))")
end

"""
    nearinteractions_consecutive(tree; args...)

Extract near-field blocks with consecutive storage layout.

Variant used when spaces are permuted to align with tree ordering. Returns
near-field block structure compatible with permuted space layout.

# Arguments

  - `tree`: Hierarchical tree object
  - `args...`: Additional keyword arguments (e.g., `isnear`)

# Returns

  - `(values, nearvalues)`: Index vectors for consecutive near-field layout

# Notes

Implemented by tree backend extensions. Must account for space permutation
applied by `PermuteSpaceInPlace()` ordering strategy.
"""
function nearinteractions_consecutive(tree; args...)
    return error("Needs to be implemented for $(typeof(tree))")
end

"""
    assemblenears(operator, testspace, trialspace, tree, ::PreserveSpaceOrder; kwargs...)

Assemble near-field blocks without reordering test and trial spaces.

Computes dense (non-low-rank) matrix blocks for all admissible cluster pairs,
retaining the original ordering of test and trial spaces. The resulting
block-sparse matrix stores these near-field interactions.

# Arguments

  - `operator`: Operator/kernel for matrix entry evaluation
  - `testspace`: Test space (row basis/evaluation points)
  - `trialspace`: Trial space (column basis/evaluation points)
  - `tree`: Hierarchical tree structure
  - `::PreserveSpaceOrder`: Space ordering strategy marker
  - `isnear`: Admissibility predicate (default: `isnear()`)
  - `scheduler`: Thread scheduler (default: `SerialScheduler()`)
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
    spaceordering::SpaceOrderingStyle;
    isnear=isnear(),
    scheduler=SerialScheduler(),
    matrixdata=defaultmatrixdata(operator, testspace, trialspace),
    verbose::Bool=true,
)
    nearmatrix = AbstractKernelMatrix(
        operator, testspace, trialspace; matrixdata=matrixdata
    )
    return assemblenears(
        nearmatrix, tree, spaceordering; isnear=isnear, scheduler=scheduler, verbose=verbose
    )
end

function assemblenears(
    nearmatrix::AbstractKernelMatrix,
    tree,
    ::PreserveSpaceOrder;
    isnear=isnear(),
    scheduler=SerialScheduler(),
    verbose::Bool=true,
)
    values, nearvalues = nearinteractions(tree; isnear=isnear)

    isempty(values) && return BlockSparseMatrix(
        Matrix{eltype(nearmatrix)}[], Vector{Int}[], Vector{Int}[], size(nearmatrix)
    )

    blocks = [
        zeros(eltype(nearmatrix), length(values[i]), length(nearvalues[i])) for
        i in eachindex(values)
    ]
    pbar = Progress(length(blocks); desc="Assembling near interactions: ", enabled=verbose)
    assemble_blocks_sparse(
        nearmatrix, blocks, values, nearvalues; scheduler=scheduler, pbar=pbar
    )
    finish!(pbar)

    return BlockSparseMatrix(blocks, values, nearvalues, size(nearmatrix))
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

Computes dense matrix blocks for admissible cluster pairs, with both spaces
permuted to align with the hierarchical tree structure. Produces a specialized
block storage format optimized for the reordered layout.

# Arguments

  - `operator`: Operator/kernel for matrix entry evaluation
  - `testspace`: Test space (row basis) — will be reordered
  - `trialspace`: Trial space (column basis) — will be reordered
  - `tree`: Hierarchical tree structure
  - `::PermuteSpaceInPlace`: Space ordering strategy marker
  - `isnear`: Admissibility predicate (default: `isnear()`)
  - `scheduler`: Thread scheduler (default: `SerialScheduler()`)
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
    nearmatrix::AbstractKernelMatrix,
    tree,
    ::PermuteSpaceInPlace;
    isnear=isnear(),
    scheduler=SerialScheduler(),
    verbose::Bool=true,
)
    values, nearvalues = nearinteractions_consecutive(tree; isnear=isnear)
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
