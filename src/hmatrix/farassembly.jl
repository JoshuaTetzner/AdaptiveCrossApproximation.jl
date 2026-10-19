"""
    assemblefars(operator, testspace, trialspace, tree, partition; compressor, maxrank,
        matrixdata, ntasks, verbose=true)

Compress the far-field blocks of the terminal blocks `partition` (see
[`interactions`](@ref)). The test clusters are processed in parallel on `ntasks`
tasks; each task owns one compressor and one pair of factor buffers, sized for the
largest far-field block, and reuses them for all its blocks.

# Returns

A `Vector{LowRankMatrix{T}}` in the order of `partition.farnodes`.
"""
function assemblefars(
    operator,
    testspace,
    trialspace,
    tree,
    partition;
    compressor=ACA(),
    maxrank::Int=40,
    matrixdata=defaultfarmatrixdata(operator, testspace, trialspace),
    ntasks::Integer=Threads.nthreads(),
    verbose::Bool=true,
)
    kernel = AbstractKernelMatrix(operator, testspace, trialspace; matrixdata=matrixdata)
    T = eltype(kernel)
    (; farptr, farnodes) = partition
    rowtree = testtree(tree)
    coltree = trialtree(tree)
    far = Vector{LowRankMatrix{T}}(undef, length(farnodes))
    isempty(far) && return far

    rownodes = [τ for τ in 1:(length(farptr) - 1) if farptr[τ + 1] > farptr[τ]]
    maxrows = maximum(τ -> numberofvalues(rowtree, τ), rownodes)
    maxcols = maximum(σ -> numberofvalues(coltree, σ), farnodes)
    entries = sum(
        numberofvalues(rowtree, τ) * numberofvalues(coltree, farnodes[k]) for τ in rownodes
        for k in farptr[τ]:(farptr[τ + 1] - 1)
    )
    progress = Progress(entries; desc="Assembling far interactions: ", enabled=verbose)
    scheduler = _scheduler(ntasks)
    @tasks for τ in rownodes
        @set scheduler = scheduler
        @local begin
            U = zeros(T, maxrows, maxrank)
            V = zeros(T, maxrank, maxcols)
            localcompressor = compressor(kernel, maxrows, maxcols, maxrank)
        end
        rows = values(rowtree, τ)
        for k in farptr[τ]:(farptr[τ + 1] - 1)
            cols = values(coltree, farnodes[k])
            Uk = fill!(view(U, 1:length(rows), :), zero(T))
            Vk = fill!(view(V, :, 1:length(cols)), zero(T))
            localmatrix = localkernelmatrix(kernel, rows, cols)
            npivots = localcompressor(
                localmatrix, Uk, Vk, maxrank; rowidcs=rows, colidcs=cols
            )
            npivots == maxrank && @warn "Maximum rank block"
            far[k] = LowRankMatrix(Uk[:, 1:npivots], Vk[1:npivots, :])
            next!(progress; step=length(rows) * length(cols))
        end
    end
    finish!(progress)
    return far
end
