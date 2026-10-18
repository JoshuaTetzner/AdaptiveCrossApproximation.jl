# Dependency graph of a task sequence in serial program order. Every task
# names the resources (block entries) it reads and writes. A task depends on
# the last earlier writer of each resource it touches and on the earlier
# readers of each resource it writes. Any execution order respecting these
# edges therefore computes exactly the serial result.
struct _TaskGraph
    succptr::Vector{Int}
    succ::Vector{Int}
    indegree::Vector{Int}
end

_ntasks(G::_TaskGraph) = length(G.indegree)

struct _TaskGraphBuilder
    lastwriter::Vector{Int}
    readers::Vector{Vector{Int}}
    deps::Vector{Vector{Int}}
end

_TaskGraphBuilder(nresources) =
    _TaskGraphBuilder(zeros(Int, nresources), [Int[] for _ in 1:nresources], Vector{Int}[])

# Append a task; `after` lists further tasks it must follow. Returns its index.
function _addtask!(b::_TaskGraphBuilder, reads, writes, after=())
    t = length(b.deps) + 1
    deps = collect(Int, after)
    for r in reads
        w = b.lastwriter[r]
        w > 0 && push!(deps, w)
    end
    for r in writes
        w = b.lastwriter[r]
        w > 0 && push!(deps, w)
        append!(deps, b.readers[r])
    end
    for r in reads
        readers = b.readers[r]
        (isempty(readers) || last(readers) != t) && push!(readers, t)
    end
    for r in writes
        b.lastwriter[r] = t
        empty!(b.readers[r])
    end
    push!(b.deps, unique!(sort!(deps)))
    return t
end

function _TaskGraph(b::_TaskGraphBuilder)
    n = length(b.deps)
    indegree = length.(b.deps)
    succptr = zeros(Int, n + 1)
    for deps in b.deps, d in deps
        succptr[d + 1] += 1
    end
    succptr[1] = 1
    cumsum!(succptr, succptr)
    succ = Vector{Int}(undef, succptr[end] - 1)
    next = succptr[1:n]
    for (t, deps) in enumerate(b.deps), d in deps
        succ[next[d]] = t
        next[d] += 1
    end
    return _TaskGraph(succptr, succ, indegree)
end

# Binary min-heap of ready tasks: the earliest task in program order first.
function _heappush!(heap::Vector{Int}, x::Int)
    push!(heap, x)
    i = length(heap)
    while i > 1
        parent = i >> 1
        heap[parent] <= x && break
        heap[i] = heap[parent]
        i = parent
    end
    heap[i] = x
    return heap
end

function _heappop!(heap::Vector{Int})
    top, x = heap[1], pop!(heap)
    n = length(heap)
    n == 0 && return top
    i = 1
    while true
        c = 2i
        c > n && break
        c < n && heap[c + 1] < heap[c] && (c += 1)
        heap[c] >= x && break
        heap[i] = heap[c]
        i = c
    end
    heap[i] = x
    return top
end

mutable struct _GraphRun
    const condition::Threads.Condition
    const ready::Vector{Int}
    const indegree::Vector{Int}
    done::Int
    stop::Bool
    error::Any
end

# Run every task of `G`, one worker per element of `workers`; `f(task, worker)`
# executes a task and returns false to stop starting new ones. Returns false if
# stopped. A single worker runs the tasks in program order; several workers
# take the earliest ready task. The first exception is rethrown after all
# workers have returned.
function _rungraph!(f::F, G::_TaskGraph, workers::AbstractVector) where {F}
    if length(workers) == 1
        worker = only(workers)
        for t in 1:_ntasks(G)
            f(t, worker) || return false
        end
        return true
    end
    indegree = copy(G.indegree)
    ready = Int[]
    for t in eachindex(indegree)
        iszero(indegree[t]) && _heappush!(ready, t)
    end
    run = _GraphRun(Threads.Condition(), ready, indegree, 0, false, nothing)
    @sync for worker in workers
        Threads.@spawn _graphworker!(f, G, run, worker)
    end
    isnothing(run.error) || throw(run.error)
    return !run.stop
end

function _graphworker!(f::F, G, run, worker) where {F}
    n = _ntasks(G)
    while true
        t = lock(run.condition) do
            while isempty(run.ready) && !run.stop && run.done < n
                wait(run.condition)
            end
            return run.stop || isempty(run.ready) ? 0 : _heappop!(run.ready)
        end
        t == 0 && return nothing
        proceed = try
            f(t, worker)
        catch error
            lock(run.condition) do
                run.stop = true
                isnothing(run.error) && (run.error = error)
                return notify(run.condition)
            end
            return nothing
        end
        lock(run.condition) do
            run.done += 1
            proceed || (run.stop = true)
            woken = 0
            for i in G.succptr[t]:(G.succptr[t + 1] - 1)
                s = G.succ[i]
                run.indegree[s] -= 1
                if iszero(run.indegree[s])
                    _heappush!(run.ready, s)
                    woken += 1
                end
            end
            if run.stop || run.done == n
                notify(run.condition)
            else
                for _ in 1:woken
                    notify(run.condition; all=false)
                end
            end
        end
    end
end

# Execution context of one worker: its scratch and the truncation status of the
# recompressions it performed.
struct _Worker{T,R}
    scratch::BlockScratch{T}
    status::_ProductStatus{R}
end

_Worker(scratch::BlockScratch{T}) where {T} =
    _Worker(scratch, _ProductStatus(true, false, zero(real(T)), 0))
_Worker{T}() where {T} = _Worker(BlockScratch{T}())

function _resetstatus!(workers)
    for worker in workers
        status = worker.status
        status.converged, status.rank_limited = true, false
        status.max_block_error, status.flushes = zero(status.max_block_error), 0
    end
    return workers
end

function _workerstatus(workers)
    statuses = map(worker -> worker.status, workers)
    return (
        converged=all(status -> status.converged, statuses),
        rank_limited=any(status -> status.rank_limited, statuses),
        max_block_error=maximum(status -> status.max_block_error, statuses),
        flushes=sum(status -> status.flushes, statuses),
    )
end

# n tasks without dependencies.
_independentgraph(n) = _TaskGraph(ones(Int, n + 1), Int[], zeros(Int, n))
