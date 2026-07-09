using AdaptiveCrossApproximation
using BEAST
using CompScienceMeshes
using H2Trees
using ParallelKMeans
using LinearAlgebra
using OhMyThreads
using Random

# Benchmark the BEAST extension's far-field ACA block-assembly path.
#
# Companion to hmatrix_pointmatrix_preserve.jl, which benchmarks HMatrix
# construction for a plain point kernel (no BEAST). This script benchmarks
# the same construction for a real BEAST operator (Maxwell EFIE on a sphere)
# to attribute time/allocations to BEAST.blockassembler, and in particular to
# BEAST.reduce_assembly_data (src/bases/basis.jl), which is suspected of
# rebuilding the same Dict{Int,Int} dof-remapping table from scratch on every
# single ACA row/column fetch, even when one side of the fetch (the full row
# or column index set of the block being compressed) does not change between
# consecutive ACA iterations.
#
# Two things are measured:
#   1. End-to-end HMatrix construction (near + far), like the point-kernel
#      benchmark, for an overall timing/allocation baseline.
#   2. A direct replay of the ACA row/column fetch pattern against
#      BEAST.blockassembler for a single real far block, isolating the
#      per-call allocation of BEAST's AssembleblockbodyFunctor and
#      extrapolating it to the full far-field assembly.
#
# Results measured with this benchmark, tracking the BEAST.blockassembler
# improvements (all in BEAST/src/integralop.jl + src/bases/basis.jl):
#   (1) 2-slot LRU caching reduced assembly data per side instead of rebuilding
#       it on every ACA pivot;
#   (2) a persistent dof->position scratch buffer instead of a Dict on every
#       cache miss;
#   (3) a shared per-test-element kernel (assemblechunk_element!) called from a
#       plain serial loop in the block assembler - instead of routing every
#       fetch through assemblechunk_body!'s @tasks/@local machinery - with
#       reused (task-local) zlocal/tadjq scratch and forced specialization of
#       the inner call (without which it dispatches dynamically and boxes all
#       its arguments on every element pair).
#   (4) an allocation-free LRU miss path: the evicted slot's els/ids/reduced-
#       data buffers are refilled in place (reduce_assembly_data! writes into
#       the slot's existing array, reallocating only when the active-element
#       count changes), and the per-fetch task_local_storage lookup no longer
#       allocates a closure, and the concrete cache type is asserted so the
#       per-fetch assembleblock! call is statically dispatched (no functor
#       boxing). This is what matters at the scales where the far field
#       dominates (50k-2M unknowns): the per-fetch cost is in the FAR path,
#       repeated billions of times.
#
# Far-field assembly only (13485 dofs, MESH_H=0.06, maxrank=40, 4 threads),
# allocation COUNT (what drives GC), before vs. after (4):
#     ~2.60 M allocations  ->  ~0.74 M allocations   (-72%)
#     @timed 0.579 GiB     ->  0.423 GiB
# The remaining far-field bytes are dominated by the fundamental LowRankMatrix
# U/V factors (and the z MVP buffer, which must stay - BlockSparseMatrices uses
# it) - i.e. the actual compressed result, not removable overhead. The 28%
# byte reduction understates the win; the 72% allocation-count drop is the GC
# story.
#
# Per-block ACA replay (one ~124x122 far block, 2*maxrank=80 row/col fetches):
#     @tasks delegation (after (1)+(2)):  ~1533 bytes/fetch
#     serial shared kernel (after (3)):    ~560 bytes/fetch
#     allocation-free miss (after (4)):    near-zero BEAST allocation/fetch
#
# End-to-end HMatrix construction (9807 dofs, 7878 far blocks, MESH_H=0.07,
# maxrank=40, 4 threads, stable/reproducible across repeated runs):
#     pristine BEAST:            3.68 GiB alloc, 0.69 s gc
#     after (1)+(2):             1.72 GiB alloc, 0.30 s gc
#     after (1)+(2)+(3):         1.61 GiB alloc, 0.25 s gc
#     after (1)+(2)+(3)+(4):     1.50 GiB alloc, 0.26 s gc
#   (this mesh is near-dominated, so the far-field count win shows up far more
#    strongly on the large, far-dominated problems this targets.)
#
# (5) AdaptiveCrossApproximation-side far-field fixes (in convergence/ and
#     aca.jl), found by profiling the PreserveSpaceOrder far path (16 threads),
#     where these dominate the *allocation count* (what drives GC):
#       - FNormEstimator convergence: normF! computed dot()/norm() over
#         @views of the col/row buffers once per ACA pivot; those SubArrays
#         escape into dot()/norm() and heap-allocate. Replaced with direct
#         indexed reductions (colnorm/rownorm/coldot/rowdot) - same arithmetic
#         (sqrt(sum(abs2)), Hermitian sum(conj(x)*y)), verified to give
#         identical pivot counts and bit-identical block errors, so the
#         convergence criterion is unchanged. This is the single biggest
#         far-field allocator by count (the cross-term loop is O(rank^2) per
#         block, so the win grows with block rank).
#       - aca.jl pivot-row normalization: `view(...) ./= view(...)` -> in-place
#         loop.
#
# Far-field assembly only, PreserveSpaceOrder, 13485 dofs, 16 threads,
# allocation COUNT, before vs. after (5):
#     ~3.50 M allocations  ->  ~1.89 M allocations   (-46%)
#     @timed 1.41 GiB      ->  0.70 GiB
# Full HMatrix construction, PreserveSpaceOrder, 51309 dofs, 16 threads, after
# all of (1)-(5): 14.3 GiB alloc, 6.2% gc.
# The largest single remaining far allocator is BlockSparseMatrices' per-level
# graph coloring (for the threaded matrix-vector product) - an intentional,
# one-time cost that buys a much faster MVP, so it is deliberately left as-is.
# The rest is the fundamental LowRankMatrix U/V/z storage.
#
# At higher thread counts (e.g. 32 threads on this 224-core machine), total
# far-field allocation becomes highly run-to-run non-deterministic (observed
# 3.68-7.44 GiB across identical repeated runs) - but this reproduces
# identically on a completely unmodified BEAST checkout, so it is a pre-existing
# characteristic of AdaptiveCrossApproximation's far-field task scheduling /
# ACA convergence under concurrency (likely floating-point non-associativity in
# parallel reductions nudging some blocks' ACA convergence checks), not
# something introduced or fixed by the changes here.
#
# Reduce MESH_H locally only for smoke tests.

const MESH_H = 0.07
const MINVALUES = 128
const TOL = 1e-3
const MAXRANK = 40
const SEED = 1234

timed_field(t, name, default=nothing) =
    name in propertynames(t) ? getproperty(t, name) : default

function print_timing(label, t)
    println(label)
    println("  elapsed seconds: ", t.time)
    println("  allocations GiB: ", t.bytes / 2.0^30)
    println("  gc seconds: ", t.gctime)
    compile_time = timed_field(t, :compile_time)
    recompile_time = timed_field(t, :recompile_time)
    lock_conflicts = timed_field(t, :lock_conflicts)
    compile_time === nothing || println("  compile seconds: ", compile_time)
    recompile_time === nothing || println("  recompile seconds: ", recompile_time)
    lock_conflicts === nothing || println("  lock conflicts: ", lock_conflicts)
    return nothing
end

function build_problem(h; minvalues=MINVALUES, seed=SEED)
    Random.seed!(seed)
    Γ = meshsphere(1.0, h)
    X = raviartthomas(Γ)
    ttree = KMeansTree(X.pos, 2; minvalues=minvalues)
    tree = BlockTree(ttree, ttree)
    return X, tree
end

function assemble_hmatrix(
    op, X, tree; tol=TOL, maxrank=MAXRANK, scheduler=DynamicScheduler()
)
    return HMatrix(
        op,
        X,
        X,
        tree;
        tol=tol,
        maxrank=maxrank,
        isnear=AdaptiveCrossApproximation.isnear(),
        scheduler=scheduler,
    )
end

function quiet_assemble_hmatrix(args...; kwargs...)
    return redirect_stderr(devnull) do
        redirect_stdout(devnull) do
            return assemble_hmatrix(args...; kwargs...)
        end
    end
end

"""
    probe_reduce_assembly_data(op, X, tree; maxrank)

Pick one real far (admissible) cluster pair out of `tree` and replay the exact
row/then/column fetch sequence ACA performs against it (see
AdaptiveCrossApproximation.jl's `ACA` callable in src/aca.jl): each ACA
iteration fetches one full row (1 x cbsize) and one full column (rbsize x 1)
of the block via `nextrc!` -> `BEASTKernelMatrix` -> `nearassembler`, i.e.
BEAST's `AssembleblockbodyFunctor`.

For a row fetch, the trial-side index set handed to
`AssembleblockbodyFunctor` is the *entire* column range of the block, unchanged
across all `maxrank` row fetches of this block's ACA loop. Likewise the
test-side index set is unchanged across all column fetches. Internally,
`AssembleblockbodyFunctor` calls `BEAST.reduce_assembly_data` on *both* sides on
*every* call, which slices a 3D array and builds a fresh `Dict{Int,Int}` dof
remapping irrespective of whether that side actually changed. This function
measures exactly that redundant cost using `@allocated`, and extrapolates it
over a full ACA run and over the whole far-field assembly.
"""
function probe_reduce_assembly_data(op, X, tree; maxrank=MAXRANK)
    values, farptr, farvalues = AdaptiveCrossApproximation.farinteractions(
        tree; isnear=AdaptiveCrossApproximation.isnear()
    )

    nodes_with_far = [
        n for n in eachindex(values) if isassigned(values, n) && farptr[n + 1] > farptr[n]
    ]
    isempty(nodes_with_far) && error("No far blocks found - mesh too small for this tree?")

    # Pick the largest block we can find, for a representative (not best-case) measurement.
    bestnode, bestfaridx, bestsize = nodes_with_far[1], farptr[nodes_with_far[1]], 0
    for n in nodes_with_far, faridx in farptr[n]:(farptr[n + 1] - 1)
        sz = length(values[n]) * length(farvalues[faridx])
        if sz > bestsize
            bestnode, bestfaridx, bestsize = n, faridx, sz
        end
    end

    rowidcs = values[bestnode]
    colidcs = farvalues[bestfaridx]
    rbsize, cbsize = length(rowidcs), length(colidcs)
    nfarblocks = farptr[end] - 1

    assembler = BEAST.blockassembler(op, X, X; quadstrat=BEAST.DoubleNumQStrat(2, 3))
    noop(v, m, n) = nothing

    # Replay a realistic ACA pivoting sequence on this single block: maxrank
    # row fetches (each against a *different* single row, like a real ACA
    # pivot) interleaved with maxrank column fetches (each against a
    # different single column), all against the *same* assembler instance so
    # any caching persists across iterations exactly as it would inside a
    # real ACA run. This is what `nextrc!` in src/aca.jl actually does.
    function replay(n; warmup=false)
        total = 0
        for it in 1:n
            p = mod1(it, rbsize)
            b = @allocated assembler(view(rowidcs, p:p), colidcs, noop)
            warmup || (total += b)
            q = mod1(it, cbsize)
            b = @allocated assembler(rowidcs, view(colidcs, q:q), noop)
            warmup || (total += b)
        end
        return total
    end

    replay(maxrank; warmup=true) # compile, and let the LRU caches settle
    bytes_total = replay(maxrank)

    println(
        "representative far block: ",
        rbsize,
        " x ",
        cbsize,
        " (out of ",
        nfarblocks,
        " far blocks total)",
    )
    println(
        "bytes for one block's full ACA run (maxrank=",
        maxrank,
        ", ",
        2maxrank,
        " row/col fetches): ",
        bytes_total,
        "  (",
        bytes_total / (2maxrank),
        " bytes/call average)",
    )
    println(
        "extrapolated over all ",
        nfarblocks,
        " far blocks: ",
        nfarblocks * bytes_total / 2.0^30,
        " GiB",
        " (assumes comparable block sizes; real blocks vary)",
    )

    return bytes_total
end

println("mesh h: ", MESH_H)
println("minvalues: ", MINVALUES)
println("tol: ", TOL)
println("maxrank: ", MAXRANK)
println("threads: ", Threads.nthreads())
println("scheduler: ", DynamicScheduler())

κ = 1.0
op = Maxwell3D.singlelayer(; wavenumber=κ)

# HMatrix's default spaceordering is PermuteSpaceInPlace(), which mutates X
# in place (permute!) as a side effect of construction. Reusing the same X
# (and the tree built around its original ordering) across two HMatrix(...)
# calls double-permutes X on the second call, desynchronizing it from tree
# and corrupting row/column alignment - so warmup and the measured run each
# get their own fresh X/tree pair.
warmup_X, warmup_tree = build_problem(MESH_H)
println("numfunctions(X): ", numfunctions(warmup_X))
GC.gc()
warmup = @timed quiet_assemble_hmatrix(op, warmup_X, warmup_tree)
print_timing("warmup construction:", warmup)

X, tree = build_problem(MESH_H)
GC.gc()
timed = @timed quiet_assemble_hmatrix(op, X, tree)
hmat = timed.value
print_timing("measured construction:", timed)

println("hmatrix size: ", size(hmat))
println("near blocks: ", length(hmat.nearinteractions.blocks))
println("far levels: ", length(hmat.farinteractions))
println("far blocks: ", sum(length(level.blocks) for level in hmat.farinteractions))
println("summary size GiB: ", Base.summarysize(hmat) / 2.0^30)

println()
println("=== reduce_assembly_data attribution probe ===")
probe_reduce_assembly_data(op, X, tree; maxrank=MAXRANK)
