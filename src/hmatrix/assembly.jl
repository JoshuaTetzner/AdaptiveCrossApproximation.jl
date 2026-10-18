"""
    terminalinteractions(tree; isnear=isnear(), spaceordering=PreserveSpaceOrder())

Return the row nodes, column nodes, near/far flags, transpose flags, and physical
storage keys for the terminal block partition. Tree backends implement the
positional method.
"""
function terminalinteractions(
    tree;
    isnear=isnear(),
    spaceordering::SpaceOrderingStyle=PreserveSpaceOrder(),
    symmetric::Bool=false,
)
    return terminalinteractions(tree, IsNearFunctor(isnear), spaceordering, Val(symmetric))
end

function terminalinteractions(
    tree, predicate::IsNearFunctor, ordering::SpaceOrderingStyle, symmetry::Val
)
    return error(
        "Terminal interactions are not implemented for $(typeof(tree)), " *
        "$(typeof(predicate)), $(typeof(ordering))",
    )
end

clusterindices(tree, node::Int, ordering::SpaceOrderingStyle) =
    error("Cluster indices are not implemented for $(typeof(tree)), $(typeof(ordering))")
numberofnodes(tree) = error("numberofnodes is not implemented for $(typeof(tree))")
_symmetrictree(tree) =
    error("A shared test/trial tree is not implemented for $(typeof(tree))")

function _assemble_hmatrix_storage(
    operator,
    testspace,
    trialspace,
    tree,
    ordering::SpaceOrderingStyle;
    compressor=ACA(),
    isnear=isnear(),
    maxrank::Int=40,
    nearmatrixdata=defaultmatrixdata(operator, testspace, trialspace),
    farmatrixdata=defaultfarmatrixdata(operator, testspace, trialspace),
    ntasks::Integer=Threads.nthreads(),
    verbose::Bool=true,
    symmetric::Bool=false,
    tol::Real=1e-4,
)
    scheduler = _scheduler(ntasks)
    rownodes, colnodes, farflags, transposed, storagekeys = terminalinteractions(
        tree; isnear=isnear, spaceordering=ordering, symmetric=symmetric
    )
    nblocks = length(rownodes)
    length(colnodes) ==
    length(farflags) ==
    length(transposed) ==
    length(storagekeys) ==
    nblocks || throw(DimensionMismatch("invalid terminal interaction metadata"))

    rowtree = testtree(tree)
    coltree = trialtree(tree)
    rowindices = [clusterindices(rowtree, node, ordering) for node in rownodes]
    colindices = [clusterindices(coltree, node, ordering) for node in colnodes]

    nearkernel = AbstractKernelMatrix(
        operator, testspace, trialspace; matrixdata=nearmatrixdata
    )
    farkernel = AbstractKernelMatrix(
        operator, testspace, trialspace; matrixdata=farmatrixdata
    )
    T = eltype(nearkernel)
    T == eltype(farkernel) || throw(ArgumentError("near and far scalar types differ"))

    firstrecords = Int[]
    seen = falses(maximum(storagekeys; init=0))
    for record in eachindex(storagekeys)
        key = storagekeys[record]
        if !seen[key]
            push!(firstrecords, record)
            seen[key] = true
        end
    end
    nearrecords = filter(record -> !farflags[record], firstrecords)
    farrecords = filter(record -> farflags[record], firstrecords)
    storageidx = zeros(Int, nblocks)
    keystorage = zeros(Int, length(seen))

    near = Matrix{T}[
        zeros(T, length(rowindices[i]), length(colindices[i])) for i in nearrecords
    ]
    for (storage, record) in enumerate(nearrecords)
        keystorage[storagekeys[record]] = storage
    end
    nearprogress = Progress(
        length(near); desc="Assembling near interactions: ", enabled=verbose
    )
    assemble_blocks(
        nearkernel,
        near,
        rowindices[nearrecords],
        colindices[nearrecords];
        ntasks=ntasks,
        pbar=nearprogress,
    )
    finish!(nearprogress)
    if symmetric
        for (storage, record) in enumerate(nearrecords)
            rownodes[record] == colnodes[record] || continue
            block = near[storage]
            for j in axes(block, 2), i in (j + 1):size(block, 1)
                block[i, j] = block[j, i]
            end
        end
    end

    far = Vector{LowRankMatrix{T}}(undef, length(farrecords))
    farprogress = Progress(
        length(far); desc="Assembling far interactions: ", enabled=verbose
    )
    @tasks for storage in eachindex(farrecords)
        @set scheduler = scheduler
        record = farrecords[storage]
        rows = rowindices[record]
        cols = colindices[record]
        U = zeros(T, length(rows), maxrank)
        V = zeros(T, maxrank, length(cols))
        localmatrix = localkernelmatrix(farkernel, rows, cols)
        localcompressor = compressor(farkernel, length(rows), length(cols), maxrank)
        npivots = localcompressor(localmatrix, U, V, maxrank; rowidcs=rows, colidcs=cols)
        npivots == maxrank && @warn "Maximum rank block"
        far[storage] = LowRankMatrix(view(U, :, 1:npivots), view(V, 1:npivots, :))
        keystorage[storagekeys[record]] = storage
        next!(farprogress)
    end
    finish!(farprogress)
    for record in eachindex(storageidx)
        storageidx[record] = keystorage[storagekeys[record]]
    end

    csr, csc = _blockindices(
        rownodes,
        colnodes,
        farflags,
        transposed,
        storageidx,
        numberofnodes(rowtree),
        numberofnodes(coltree),
    )
    return HMatrix{T}(
        tree, near, far, csr, csc, ordering, (length(testspace), length(trialspace)), tol
    )
end
