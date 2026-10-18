# Matrix fields are contiguous owned arrays. Sort their byte intervals once in
# preflight, rather than testing every source against every destination block.
# Source/source sharing is allowed; destination/destination and cross-view
# sharing are rejected.
function _pushworkspacespans!(spans, W, destination)
    for A in W.near
        isempty(A) ||
            push!(spans, (UInt(pointer(A)), UInt(pointer(A)) + sizeof(A), destination))
    end
    for R in W.far, A in (R.U, R.V)
        isempty(A) ||
            push!(spans, (UInt(pointer(A)), UInt(pointer(A)) + sizeof(A), destination))
    end
    return spans
end

function _checkworkspacealias(C, A, scratch)
    spans = Tuple{UInt,UInt,Bool}[]
    GC.@preserve C A begin
        _pushworkspacespans!(spans, C, true)
        _bound(C, A) || _pushworkspacespans!(spans, A, false)
        sort!(spans)
        destend, sourceend = zero(UInt), zero(UInt)
        for (start, stop, destination) in spans
            if destination
                start >= max(destend, sourceend) || throw(
                    ArgumentError(
                        "destination storage overlaps another block or the source"
                    ),
                )
                destend = max(destend, stop)
            else
                start >= destend ||
                    throw(ArgumentError("source and destination matrices overlap"))
                sourceend = max(sourceend, stop)
            end
        end
    end
    for W in (C, A), buffer in _scratcharrays(scratch)
        for block in W.near
            _checkproductalias(buffer, block)
        end
        for block in W.far
            _checkproductalias(buffer, block)
        end
    end
    return nothing
end

_bound(C::_TreeView, A::_TreeView) =
    length(C.near) == length(A.near) &&
    length(C.far) == length(A.far) &&
    all(i -> C.near[i] === A.near[i], eachindex(C.near)) &&
    all(i -> C.far[i] === A.far[i], eachindex(C.far))

function _checkadditionpartition(C, A)
    size(C) == size(A) || throw(DimensionMismatch("matrix dimensions differ"))
    testtree(C.tree) === testtree(A.tree) && trialtree(C.tree) === trialtree(A.tree) ||
        throw(ArgumentError("addition requires the same row and column cluster trees"))
    c, a = C.csr, A.csr
    c === a ||
        c.rowptr == a.rowptr && c.colnodes == a.colnodes && c.isfar == a.isfar ||
        throw(
            ArgumentError("addition requires identical terminal partitions and block kinds")
        )
    for k in eachindex(c.storageidx)
        _checkworkspaceblock(C, k)
        _checkworkspaceblock(A, k)
    end
    return nothing
end

# C += alpha*A for views with identical partitions; physical storage orders may
# differ. Dense blocks are updated exactly; every far block is recompressed once,
# the blocks distributed over `workers`. `C` and `A` may be bound to the same
# matrix. Returns the aggregated local truncation status. Structural, scalar and
# alias errors are checked before writes.
function _hierarchicaladd!(
    C::_TreeView{T}, A::_TreeView{T}, coefficient, workers::AbstractVector, options
) where {T}
    # A fresh name: closures box reassigned captured variables.
    alpha = convert(T, coefficient)
    isfinite(alpha) || throw(ArgumentError("alpha must be finite"))
    same = _bound(C, A)
    same && !isfinite(one(T) + alpha) && throw(ArgumentError("1 + alpha must be finite"))
    _checkadditionpartition(C, A)
    _checkworkspacealias(C, A, first(workers).scratch)
    _resetstatus!(workers)
    if same
        # C += alpha*C is an exact scaling.
        foreach(D -> rmul!(D, one(T) + alpha), C.near)
        foreach(R -> _scalelowrank!(R, one(T) + alpha), C.far)
    elseif !iszero(alpha)
        c, a = C.csr, A.csr
        _rungraph!(_independentgraph(length(c.storageidx)), workers) do k, worker
            if c.isfar[k]
                R, S = C.far[c.storageidx[k]], A.far[a.storageidx[k]]
                info = _addlowrank!(R, S, alpha, worker.scratch; options...)
                status = worker.status
                status.converged &= info.converged
                status.rank_limited |= info.rank_limited
                status.max_block_error = max(status.max_block_error, info.error)
            else
                axpy!(alpha, A.near[a.storageidx[k]], C.near[c.storageidx[k]])
            end
            return true
        end
    end
    total = _workerstatus(workers)
    return (
        converged=total.converged,
        rank_limited=total.rank_limited,
        max_block_error=total.max_block_error,
    )
end

_hierarchicaladd!(C::_TreeView, A::_TreeView, alpha, scratch::BlockScratch, options) =
    _hierarchicaladd!(C, A, alpha, [_Worker(scratch)], options)
