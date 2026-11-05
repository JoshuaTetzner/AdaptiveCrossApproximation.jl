# Far field: one prepared compressor and factor/pivot buffers sized for the largest
# far-field block, as in `assemblefars`.
function HFactory.farscratch(far::LowRankCompression, A, index)
    (; test, trial, csr) = index
    maxrows = maxcols = 0
    for row in eachindex(test.ranges), k in csr.rowptr[row]:(csr.rowptr[row + 1] - 1)
        csr.isfar[k] || continue
        maxrows = max(maxrows, length(test.ranges[row]))
        maxcols = max(maxcols, length(HFactory.clusterrange(trial, csr.colnodes[k])))
    end
    T = eltype(A)
    return (;
        compressor=far.compressor(A, maxrows, maxcols, far.maxrank),
        U=zeros(T, maxrows, far.maxrank),
        V=zeros(T, far.maxrank, maxcols),
        rows=zeros(Int, far.maxrank),
        cols=zeros(Int, far.maxrank),
    )
end

HFactory.compressedblocktype(::LowRankCompression, A) = LowRankMatrix{eltype(A)}

function HFactory.assemble_far!(
    A, index::HFactory.BlockIndex, rownode, colnode, far::LowRankCompression, scratch
)
    rows, cols = HFactory.blockindices(index, rownode, colnode)
    T = eltype(A)
    U = fill!(view(scratch.U, 1:length(rows), :), zero(T))
    V = fill!(view(scratch.V, :, 1:length(cols)), zero(T))
    r = scratch.compressor(
        localkernelmatrix(A, rows, cols),
        U,
        V,
        far.maxrank;
        rows=scratch.rows,
        cols=scratch.cols,
        rowidcs=rows,
        colidcs=cols,
    )
    r == far.maxrank && @warn "Maximum rank block" maxlog = 1
    # Copies the factors out of the reusable buffers.
    return LowRankMatrix(view(U, :, 1:r), view(V, 1:r, :))
end
