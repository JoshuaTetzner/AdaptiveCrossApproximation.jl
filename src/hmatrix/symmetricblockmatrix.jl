# Compatibility with BlockSparseMatrices 0.3.1: serial MVPs need no conflict graphs.
# Remove this constructor bypass when the public constructor supports that case.
function _serialsymmetricblockmatrix(
    diagonals::Vector{Matrix{T}},
    diagindices::Vector{Vector{Int}},
    offdiagonals::Vector{M},
    rowindices::Vector{Vector{Int}},
    colindices::Vector{Vector{Int}},
    dim::Tuple{Int,Int},
    scheduler::SerialScheduler,
) where {T,M}
    diagcolors = [collect(eachindex(diagonals))]
    offcolors = [collect(eachindex(offdiagonals))]
    return SymmetricBlockMatrix{T,Matrix{T},Int,M,typeof(scheduler)}(
        diagonals,
        diagindices,
        offdiagonals,
        rowindices,
        colindices,
        dim,
        diagcolors,
        offcolors,
        offcolors,
        scheduler,
    )
end
