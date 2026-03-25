function collectassigned(v::Vector{B}) where {B}
    nassigned = 0
    @inbounds for i in eachindex(v)
        nassigned += isassigned(v, i)
    end

    compact = Vector{B}(undef, nassigned)
    k = 0
    @inbounds for i in eachindex(v)
        if isassigned(v, i)
            k += 1
            compact[k] = v[i]
        end
    end

    return compact
end

function collectnears(vals::B, nearvals::Vector{B}, assigned::Vector{Bool}) where {B}
    nassigned = 0
    @inbounds for ass in assigned
        ass && (nassigned += 1)
    end

    compactvals = B(undef, nassigned)
    compactnearvals = Vector{B}(undef, nassigned)
    k = 0
    @inbounds for i in eachindex(vals)
        if assigned[i]
            k += 1
            compactvals[k] = vals[i]
            compactnearvals[k] = nearvals[i]
        end
    end

    return compactvals, compactnearvals
end

function linearizestorage(v::Vector{Vector{T}}) where {T}
    ptr = Vector{Int}(undef, length(v) + 1)
    ptr[1] = 1

    @inbounds for i in eachindex(v)
        ptr[i + 1] = ptr[i] + length(v[i])
    end

    data = Vector{T}(undef, ptr[end] - 1)

    @inbounds for i in eachindex(v)
        vi = v[i]
        copyto!(data, ptr[i], vi, 1, length(vi))
    end

    return ptr, data
end

function buffersize(
    vals::Vector{T}, ptr::Vector{Int}, farvals::Vector{T}, nodes::Vector{Int}
) where {T}
    blen = 0
    fblen = 0
    @inbounds for node in nodes
        start = ptr[node]
        stop  = ptr[node + 1] - 1
        start > stop && continue
        len = length(vals[node])
        blen < len && (blen = len)
        for faridx in start:stop
            len = length(farvals[faridx])
            if len > fblen
                fblen = len
            end
        end
    end
    return blen, fblen
end

# required for blocksparse format
function blockvalues(
    values::Vector{Vector{Int}},
    farptr::Vector{Int},
    farvalues::Vector{Vector{Int}},
    levelnodes::Vector{Int},
)
    nblocks = 0
    @inbounds for node in levelnodes
        nblocks += farptr[node + 1] - farptr[node]
    end
    levelvals = Vector{Vector{Int}}(undef, nblocks)
    levelfarvals = Vector{Vector{Int}}(undef, nblocks)
    i = 1
    for node in levelnodes
        farptr[node + 1] == farptr[node] && continue
        r = values[node]
        for j in farptr[node]:(farptr[node + 1] - 1)
            levelfarvals[i] = farvalues[j]
            levelvals[i] = r
            i += 1
        end
    end

    return levelvals, levelfarvals
end
