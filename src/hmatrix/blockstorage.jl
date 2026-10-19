"""
    BlockCSR{I}

Canonical row-wise (CSR) index of the terminal blocks of an H-matrix. The
terminals of row cluster `τ` are the entries `rowptr[τ]:(rowptr[τ + 1] - 1)`,
sorted by column cluster `colnodes[k]`. `isfar[k]` selects the storage vector
(`far` or `near`) and `storageidx[k]` the position of the block in it.
`transposed[k]` marks an entry that applies the transpose of a block stored for
the opposite orientation (symmetric storage only).
"""
struct BlockCSR{I<:Integer}
    rowptr::Vector{I}
    colnodes::Vector{I}
    isfar::BitVector
    transposed::BitVector
    storageidx::Vector{I}
end

"""
    BlockCSC{I}

Column-wise (CSC) traversal of the terminal blocks. The terminals of column
cluster `σ` are `colptr[σ]:(colptr[σ + 1] - 1)`, with row clusters `rownodes[j]`;
`csrmap[j]` points into the canonical [`BlockCSR`](@ref) entries, so no block is
stored twice.
"""
struct BlockCSC{I<:Integer}
    colptr::Vector{I}
    rownodes::Vector{I}
    csrmap::Vector{I}
end

Base.length(index::BlockCSR) = length(index.colnodes)
Base.length(index::BlockCSC) = length(index.rownodes)

function _compressedptr(nodes::AbstractVector{<:Integer}, nnodes::Int)
    counts = zeros(Int, nnodes)
    for node in nodes
        1 <= node <= nnodes || throw(BoundsError(1:nnodes, node))
        counts[node] += 1
    end
    ptr = Vector{Int}(undef, nnodes + 1)
    ptr[1] = 1
    for node in 1:nnodes
        ptr[node + 1] = ptr[node] + counts[node]
    end
    return ptr
end

function _blockindices(
    rownodes::Vector{Int},
    colnodes::Vector{Int},
    isfar::BitVector,
    transposed::BitVector,
    storageidx::Vector{Int},
    nrownodes::Int,
    ncolnodes::Int,
)
    nblocks = length(rownodes)
    length(colnodes) ==
    length(isfar) ==
    length(transposed) ==
    length(storageidx) ==
    nblocks || throw(DimensionMismatch("terminal block metadata has inconsistent lengths"))

    csrperm = sortperm(eachindex(rownodes); by=i -> (rownodes[i], colnodes[i]))
    csrrows = rownodes[csrperm]
    csrcols = colnodes[csrperm]
    csrfar = isfar[csrperm]
    csrtransposed = transposed[csrperm]
    csrstorage = storageidx[csrperm]

    for i in 2:nblocks
        (csrrows[i], csrcols[i]) == (csrrows[i - 1], csrcols[i - 1]) &&
            throw(ArgumentError("duplicate terminal block ($(csrrows[i]), $(csrcols[i]))"))
    end

    csr = BlockCSR(
        _compressedptr(csrrows, nrownodes),
        csrcols,
        BitVector(csrfar),
        BitVector(csrtransposed),
        csrstorage,
    )

    cscperm = sortperm(eachindex(csrrows); by=i -> (csrcols[i], csrrows[i]))
    csc = BlockCSC(_compressedptr(csrcols[cscperm], ncolnodes), csrrows[cscperm], cscperm)
    return csr, csc
end

"""
    blockindices(tree, partition)

Block index `(csr, csc)` of the terminal blocks `partition` of
[`interactions`](@ref): the `k`-th near block refers to `near[k]`, the `k`-th far
block to `far[k]`.
"""
function blockindices(tree, partition)
    (; nearptr, nearnodes, farptr, farnodes) = partition
    nnear, nfar = length(nearnodes), length(farnodes)
    return _blockindices(
        [_rownodes(nearptr); _rownodes(farptr)],
        [nearnodes; farnodes],
        [falses(nnear); trues(nfar)],
        falses(nnear + nfar),
        [1:nnear; 1:nfar],
        numberofnodes(testtree(tree)),
        numberofnodes(trialtree(tree)),
    )
end

"""
    symmetricblockindices(tree, partition)

Block index `(csr, csc)` of the transpose-symmetric terminal blocks `partition` of
[`symmetricinteractions`](@ref): every stored pair `(a, b)` and, for `a != b`, the
opposite block `(b, a)` as the transpose of the same stored block.
"""
function symmetricblockindices(tree, partition)
    (; nearptr, nearnodes, farptr, farnodes) = partition
    nnear, nfar = length(nearnodes), length(farnodes)
    rownodes = [_rownodes(nearptr); _rownodes(farptr)]
    colnodes = [nearnodes; farnodes]
    isfar = [falses(nnear); trues(nfar)]
    storageidx = [1:nnear; 1:nfar]
    opposite = findall(rownodes .!= colnodes)
    nnodes = numberofnodes(testtree(tree))
    return _blockindices(
        [rownodes; colnodes[opposite]],
        [colnodes; rownodes[opposite]],
        [isfar; isfar[opposite]],
        [falses(nnear + nfar); trues(length(opposite))],
        [storageidx; storageidx[opposite]],
        nnodes,
        nnodes,
    )
end
