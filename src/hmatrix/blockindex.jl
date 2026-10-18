"""
Canonical CSR index of terminal hierarchical blocks.
"""
struct BlockCSR{I<:Integer}
    rowptr::Vector{I}
    colnodes::Vector{I}
    isfar::BitVector
    transposed::BitVector
    storageidx::Vector{I}
end

"""
CSC traversal index. `csrmap` points into the canonical CSR entries.
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
