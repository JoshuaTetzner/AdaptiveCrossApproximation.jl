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

function buffersize(ptr::Vector{Int}, vals::Vector{T}, nodes::Vector{Int}) where {T}
    blen = 0
    @inbounds for node in nodes
        start = ptr[node]
        stop  = ptr[node + 1] - 1
        for faridx in start:stop
            len = length(vals[faridx])
            if len > blen
                blen = len
            end
        end
    end
    return blen
end
