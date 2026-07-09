using AdaptiveCrossApproximation
using H2Trees
using LinearAlgebra
using OhMyThreads
using ParallelKMeans
using Random
using StaticArrays

# Benchmark the complete HMatrix construction path on a point-kernel matrix.
#
# This intentionally uses PreserveSpaceOrder() so construction goes through the
# non-permuted BlockSparseMatrix path. The benchmark is meant as a baseline for
# ACA/HMatrix overhead without BEAST block-assembly costs.
#
# The baseline problem uses N = 50_000. Reduce this locally only for smoke tests.
# Record measured timings and lock conflict counts below when running on the
# target machine.
#
# Performance notes:
# - 2026-06-22, Julia --threads=16, workspace environment:
#   N = 50_000, DynamicScheduler(), PreserveSpaceOrder()
#   full-size warmup: 92.39 s, 21.18 GiB allocated, 3.39 s GC, 0 lock conflicts
#   measured: 91.67 s, 21.03 GiB allocated, 3.17 s GC, 0 lock conflicts
# - After FNormEstimator norm reuse and MaximumValue scan cleanup:
#   measured: 89.05 s, 20.04 GiB allocated, 2.96 s GC, 0 lock conflicts

const N = 50_000
const WARMUP_N = 1_000
const MINVALUES = 128
const TOL = 1e-4
const MAXRANK = 40
const SEED = 1234
const FULL_SIZE_WARMUP = true

struct InverseDistanceKernel{T} end

Base.eltype(::InverseDistanceKernel{T}) where {T} = T

function (::InverseDistanceKernel{T})(x, y) where {T}
    r = norm(x - y)
    return iszero(r) ? zero(T) : inv(r)
end

timed_field(t, name, default=nothing) =
    name in propertynames(t) ? getproperty(t, name) : default

function build_problem(n; minvalues=MINVALUES, seed=SEED)
    Random.seed!(seed)
    points = [@SVector rand(3) for _ in 1:n]
    cluster_tree = KMeansTree(points, 2; minvalues=minvalues)
    tree = BlockTree(cluster_tree, cluster_tree)
    return points, tree
end

function assemble_hmatrix(
    points, tree; tol=TOL, maxrank=MAXRANK, scheduler=DynamicScheduler()
)
    kernel = InverseDistanceKernel{Float64}()
    return HMatrix(
        kernel,
        points,
        points,
        tree;
        tol=tol,
        maxrank=maxrank,
        spaceordering=AdaptiveCrossApproximation.PreserveSpaceOrder(),
        scheduler=scheduler,
    )
end

function quiet_assemble_hmatrix(args...; kwargs...)
    # HMatrix currently contains internal @time calls for near/far assembly.
    # Suppress those here so this benchmark controls the output.
    return redirect_stderr(devnull) do
        redirect_stdout(devnull) do
            return assemble_hmatrix(args...; kwargs...)
        end
    end
end

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

println("n: ", N)
println("warmup n: ", WARMUP_N)
println("minvalues: ", MINVALUES)
println("tol: ", TOL)
println("maxrank: ", MAXRANK)
println("threads: ", Threads.nthreads())
println("scheduler: ", DynamicScheduler())
println("spaceordering: PreserveSpaceOrder()")
println("full size warmup: ", FULL_SIZE_WARMUP)

warmup_points, warmup_tree = build_problem(WARMUP_N)
GC.gc()
warmup = @timed quiet_assemble_hmatrix(warmup_points, warmup_tree)
print_timing("warmup construction:", warmup)

points, tree = build_problem(N)
if FULL_SIZE_WARMUP
    GC.gc()
    full_warmup = @timed quiet_assemble_hmatrix(points, tree)
    print_timing("full-size warmup construction:", full_warmup)
end

GC.gc()
timed = @timed quiet_assemble_hmatrix(points, tree)
hmat = timed.value

print_timing("measured construction:", timed)

# Keep the matrix alive and report a tiny amount of structure without forcing
# Matrix(hmat), which would destroy the benchmark's scale.
println("hmatrix size: ", size(hmat))
println("near blocks: ", length(hmat.nearinteractions.blocks))
println("far levels: ", length(hmat.farinteractions))
println("far blocks: ", sum(length(level.blocks) for level in hmat.farinteractions))
println("summary size GiB: ", Base.summarysize(hmat) / 2.0^30)
