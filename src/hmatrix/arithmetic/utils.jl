_overlap(a, b) = max(first(a), first(b)):min(last(a), last(b))
_relative(a, origin) = (first(a) - first(origin) + 1):(last(a) - first(origin) + 1)

# Equal-cost contiguous chunks of `costs`, as pointers into it.
function _costchunks(costs, nchunks)
    total = sum(costs; init=0.0)
    ptr, acc, c = [1], 0.0, 1
    for (i, cost) in enumerate(costs)
        acc += cost
        if c < nchunks && acc >= total * c / nchunks
            push!(ptr, i + 1)
            c += 1
        end
    end
    ptr[end] == length(costs) + 1 || push!(ptr, length(costs) + 1)
    return ptr
end

# Workspaces analysed for one core may serve every core with the same tree and
# terminal partition; numerical ranks may differ.
function _checkstructure(index, core::HMatrixCore)
    size(core) == index.dim || throw(DimensionMismatch("matrix dimensions differ"))
    core.tree === index.tree ||
        throw(ArgumentError("matrix uses a different cluster tree than the workspace"))
    a, b = core.csr, index.csr
    a === b ||
        a.rowptr == b.rowptr &&
        a.colnodes == b.colnodes &&
        a.isfar == b.isfar &&
        a.transposed == b.transposed &&
        a.storageidx == b.storageidx ||
        throw(ArgumentError("matrix has a different block partition than the workspace"))
    return nothing
end
