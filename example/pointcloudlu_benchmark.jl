# Run from an environment containing ACA, H2Trees, HMatrices, ParallelKMeans
# and StaticArrays. Example:
# julia --project=/tmp/aca-hmatrix-comparison --threads=32 \
#   example/pointcloudlu_benchmark.jl 2000 10000 50000
using AdaptiveCrossApproximation
using HMatrices: HMatrices
using H2Trees
using LinearAlgebra
using ParallelKMeans
using Printf
using Random
using StaticArrays
using TOML

const PCACA = AdaptiveCrossApproximation

struct CoulombBenchmarkKernel
    radius::Float64
    diagonal::Float64
    offset::Float64
    scale::Float64
end
CoulombBenchmarkKernel(radius, diagonal) = CoulombBenchmarkKernel(radius, diagonal, 0, 1)
Base.eltype(::CoulombBenchmarkKernel) = Float64
@inline function (k::CoulombBenchmarkKernel)(x, y)
    r2 = sum(abs2, x - y)
    iszero(r2) && iszero(k.radius) && iszero(k.offset) && return k.diagonal
    return k.scale / (sqrt(r2 + k.radius^2) + k.offset)
end

function pc_points(rng, n, geometry)
    geometry === :cube && return [SVector{3,Float64}(rand(rng, 3)) for _ in 1:n]
    geometry === :sphere || throw(ArgumentError("geometry must be :cube or :sphere"))
    # Match HMatrices' published example, including uniform polar angle.
    return [
        begin
            theta, phi = pi * rand(rng), 2pi * rand(rng)
            SVector(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta))
        end for _ in 1:n
    ]
end

function pc_dense(points, kernel)
    n = length(points)
    A = Matrix{Float64}(undef, n, n)
    Threads.@threads :static for j in 1:n
        @inbounds for i in 1:n
            A[i, j] = kernel(points[i], points[j])
        end
    end
    return A
end

function pc_measure(f, label)
    GC.gc()
    println("Starting ", label)
    flush(stdout)
    result = @timed f()
    @printf(
        "%s: %.6f s; %.3f GiB allocated; %.3f s GC\n",
        label,
        result.time,
        result.bytes / 2.0^30,
        result.gctime
    )
    flush(stdout)
    return result.value,
    Dict{String,Any}(
        "seconds" => result.time,
        "allocated_bytes" => result.bytes,
        "gc_seconds" => result.gctime,
    )
end

function pc_factor(f, label)
    started = time_ns()
    try
        F, timing = pc_measure(f, label)
        metrics = Dict{String,Any}(timing)
        metrics["status"] = "ok"
        return F, metrics
    catch error
        error isa SingularException ||
            error isa ZeroPivotException ||
            error isa PCACA.LeafLUFailure ||
            rethrow()
        message = sprint(showerror, error)
        println(label, " failed: ", message)
        flush(stdout)
        return nothing,
        Dict{String,Any}(
            "status" => "failed",
            "error" => message,
            "seconds" => (time_ns() - started) / 1e9,
        )
    end
end

# Matched partitions. HMatrices makes every block with a leaf cluster dense, so
# both packages use that rule together with one admissibility test on one shared
# binary KMeans tree. `:box` is HMatrices' StrongAdmissibilityStd on the shared
# tree's bounding boxes; `:ball` is ACA's bounding-ball criterion.
struct PCLeafDenseBall
    eta::Float64
end
function (p::PCLeafDenseBall)(ta, tb, a, b)
    (H2Trees.isleaf(ta, a) || H2Trees.isleaf(tb, b)) && return true
    ra, rb = H2Trees.radius(ta, a), H2Trees.radius(tb, b)
    gap = norm(H2Trees.center(ta, a) - H2Trees.center(tb, b)) - (ra + rb)
    return !(2 * max(ra, rb) <= p.eta * max(gap, 0.0))
end

struct PCLeafDenseBox{P}
    lo::Vector{P}
    hi::Vector{P}
    eta::Float64
end
function PCLeafDenseBox(tree, points, eta)
    lo = [
        reduce((x, y) -> min.(x, y), view(points, H2Trees.values(tree, node))) for
        node in eachindex(tree.nodes)
    ]
    hi = [
        reduce((x, y) -> max.(x, y), view(points, H2Trees.values(tree, node))) for
        node in eachindex(tree.nodes)
    ]
    return PCLeafDenseBox(lo, hi, eta)
end
function (p::PCLeafDenseBox)(ta, tb, a, b)
    (H2Trees.isleaf(ta, a) || H2Trees.isleaf(tb, b)) && return true
    diameter = min(norm(p.hi[a] - p.lo[a]), norm(p.hi[b] - p.lo[b]))
    gap = norm(max.(0, p.lo[a] .- p.hi[b]) .+ max.(0, p.lo[b] .- p.hi[a]))
    return !(diameter < p.eta * gap)
end

# The shared tree as an HMatrices ClusterTree with identical nodes, children order
# and permutation, plus a lookup from HMatrices nodes to H2Trees node ids.
function pc_mirror(tree, points)
    loc2glob = H2Trees.values(tree, H2Trees.root(tree))
    elements = points[loc2glob]
    glob2loc = invperm(loc2glob)
    ids = IdDict{Any,Int}()
    function build(node, start, parent)
        range = start:(start + H2Trees.numberofvalues(tree, node) - 1)
        box = HMatrices.HyperRectangle(
            reduce((x, y) -> min.(x, y), view(elements, range)),
            reduce((x, y) -> max.(x, y), view(elements, range)),
        )
        clt = HMatrices.ClusterTree(
            elements, box, range, loc2glob, glob2loc, nothing, parent
        )
        ids[clt] = node
        for child in H2Trees.children(tree, node)
            push!(clt.children, build(child, start, clt))
            start += H2Trees.numberofvalues(tree, child)
        end
        return clt
    end
    return build(H2Trees.root(tree), 1, nothing), ids
end

function pc_blocks(H::PCACA.HMatrixCore)
    index = PCACA._TreeIndex(H)
    return Set(
        (PCACA._rowrange(index, k), PCACA._colrange(index, k), H.csr.isfar[k]) for
        k in eachindex(H.csr.storageidx)
    )
end
pc_blocks(H::HMatrices.HMatrix) = Set(
    (HMatrices.rowrange(b), HMatrices.colrange(b), !(HMatrices.data(b) isa Matrix)) for
    b in HMatrices.leaves(H)
)

pc_rankhistogram(ranks) = [
    count(r -> lo < r <= hi, ranks) for
    (lo, hi) in ((0, 8), (8, 16), (16, 32), (32, 64), (64, typemax(Int)))
]

pc_payload(W) =
    sum(sizeof, W.near; init=0) + sum(R -> sizeof(R.U) + sizeof(R.V), W.far; init=0)

function pc_hmstats(H)
    blocks = HMatrices.leaves(H)
    payload, maxrank, nfar, nnear = 0, 0, 0, 0
    for block in blocks
        d = HMatrices.data(block)
        if d isa Matrix
            payload += sizeof(d)
            nnear += 1
        else
            payload += sizeof(d.A) + sizeof(d.B)
            maxrank = max(maxrank, rank(d))
            nfar += 1
        end
    end
    return Dict(
        "payload_bytes" => payload,
        "maxrank" => maxrank,
        "far_blocks" => nfar,
        "near_blocks" => nnear,
        "rank_histogram" => pc_rankhistogram([
            rank(HMatrices.data(b)) for b in blocks if !(HMatrices.data(b) isa Matrix)
        ]),
    )
end

function pc_acastats(W)
    return Dict(
        "payload_bytes" => pc_payload(W),
        "maxrank" => maximum(rank, W.far; init=0),
        "far_blocks" => length(W.far),
        "near_blocks" => length(W.near),
        "rank_histogram" => pc_rankhistogram(map(rank, W.far)),
    )
end

function pc_case(
    n;
    tol=1e-6,
    leafsize=64,
    maxrank=256,
    eta=3.0,
    hm_eta=eta,
    radius_update=H2Trees.boundingsphere,
    partition=:box,
    radius=0.0,
    diagonal=0.0,
    offset=1e-8,
    scale=inv(4pi),
    geometry=:sphere,
    seed=87123,
    hierarchical_blas=1,
    dense_blas=32,
    samples=1,
    batchrank=128,
)
    println(
        "\nPOINT CLOUD n=$n; tol=$tol; leafsize=$leafsize; geometry=$geometry; radius=$radius; offset=$offset; diagonal=$diagonal",
    )
    rng = MersenneTwister(seed)
    points = pc_points(rng, n, geometry)
    kernel = CoulombBenchmarkKernel(radius, diagonal, offset, scale)
    xtrue = randn(rng, n)
    probe = randn(rng, n)
    result = Dict{String,Any}(
        "n" => n,
        "tol" => tol,
        "leafsize" => leafsize,
        "assembly_rank_cap" => maxrank,
        "eta" => eta,
        "hm_eta" => hm_eta,
        "radius_update" => string(radius_update),
        "radius" => radius,
        "diagonal" => diagonal,
        "offset" => offset,
        "scale" => scale,
        "self_interaction" => kernel(first(points), first(points)),
        "seed" => seed,
        "samples" => samples,
        "batchrank" => batchrank,
        "julia_threads" => Threads.nthreads(),
        "hierarchical_blas_threads" => hierarchical_blas,
        "dense_blas_threads" => dense_blas,
        "julia_version" => string(VERSION),
        "hm_version" => string(pkgversion(HMatrices)),
        "blas" => string(BLAS.get_config()),
        "cpu" => Sys.CPU_NAME,
        "geometry" => string(geometry),
        "partition" => string(partition),
    )
    partition in (:box, :ball, :native) ||
        throw(ArgumentError("partition must be :box, :ball or :native"))

    BLAS.set_num_threads(dense_blas)
    A, result["dense_assembly"] = pc_measure(
        () -> pc_dense(points, kernel), "dense assembly"
    )
    rhs, reference_probe = A * xtrue, A * probe
    diagonal_value = kernel(first(points), first(points))
    offdiagonal_probe_norm = norm(reference_probe - diagonal_value * probe)
    dense_solve = nothing
    for sample in 1:samples
        D, copytime = pc_measure(() -> copy(A), "dense input copy $sample")
        F, factime = pc_measure(() -> lu!(D), "dense LU $sample")
        solution = copy(rhs)
        _, solvetime = pc_measure(() -> ldiv!(F, solution), "dense solve $sample")
        factime["solve_residual"] = norm(A * solution - rhs) / norm(rhs)
        factime["solution_error"] = norm(solution - xtrue) / norm(xtrue)
        factime["payload_bytes"] = sizeof(D)
        result["dense_input_copy_$sample"] = copytime
        result["dense_lu_$sample"] = factime
        result["dense_solve_$sample"] = solvetime
        dense_solve = solution
    end
    GC.gc()

    BLAS.set_num_threads(hierarchical_blas)
    cluster, result["aca_tree"] = pc_measure("ACA KMeansTree") do
        return H2Trees.KMeansTree(
            points;
            builder=H2Trees.KMeansTreeBuilder(;
                numberofclusters=2,
                minvalues=leafsize,
                updateradii=radius_update,
                splitterkwargs=(; rng=MersenneTwister(seed), n_threads=Threads.nthreads()),
            ),
        )
    end
    # :native keeps each package's own admissibility (and tree, below).
    predicate = if partition === :box
        PCLeafDenseBox(cluster, points, eta)
    elseif partition === :ball
        PCLeafDenseBall(eta)
    else
        eta
    end
    H, result["aca_assembly"] = pc_measure("ACA assembly") do
        return PCACA.HMatrix(
            kernel,
            points,
            points,
            H2Trees.BlockTree(cluster, cluster);
            tol,
            maxrank,
            isnear=PCACA.isnear(predicate),
            verbose=false,
        )
    end
    merge!(result["aca_assembly"], pc_acastats(H))
    result["aca_assembly"]["probe_error"] =
        norm(H * probe - reference_probe) / norm(reference_probe)
    result["aca_assembly"]["offdiagonal_probe_error"] =
        result["aca_assembly"]["probe_error"] * norm(reference_probe) /
        offdiagonal_probe_norm
    # One LU workspace serves every sample: lu! on a fresh copy of H reuses it.
    workspace, result["aca_plan"] = pc_measure(
        () -> PCACA.LUWorkspace(H; batchrank), "ACA LU planning"
    )
    for sample in 1:samples
        W, result["aca_input_copy_$sample"] = pc_measure(
            () -> copy(H), "ACA input copy $sample"
        )
        F, factime = pc_factor(() -> lu!(W, workspace), "ACA LU $sample")
        if isnothing(F)
            result["aca_lu_$sample"] = factime
            break
        end
        solver = PCACA.SolveWorkspace(F)
        solution = copy(rhs)
        _, solvetime = pc_measure(() -> ldiv!(solver, solution), "ACA solve $sample")
        BLAS.set_num_threads(dense_blas)
        factime["solve_residual"] = norm(A * solution - rhs) / norm(rhs)
        factime["solution_error"] = norm(solution - xtrue) / norm(xtrue)
        factime["dense_solution_error"] = norm(solution - dense_solve) / norm(dense_solve)
        BLAS.set_num_threads(hierarchical_blas)
        merge!(factime, pc_acastats(W))
        factime["converged"] = F.converged
        factime["rank_limited"] = F.rank_limited
        factime["flushes"] = F.flushes
        result["aca_lu_$sample"] = factime
        result["aca_solve_$sample"] = solvetime
        println(
            "ACA accuracy: residual=",
            factime["solve_residual"],
            "; solution error=",
            factime["solution_error"],
        )
    end

    hcluster, result["hm_tree"] = pc_measure("HMatrices ClusterTree") do
        partition === :native || return pc_mirror(cluster, points)
        return HMatrices.ClusterTree(
            points, HMatrices.GeometricSplitter(; nmax=leafsize); threads=true
        ),
        nothing
    end
    hcluster, hids = hcluster
    adm = if partition === :native
        HMatrices.StrongAdmissibilityStd(hm_eta)
    else
        (X, Y) -> !predicate(cluster, cluster, hids[X], hids[Y])
    end
    K = HMatrices.KernelMatrix(kernel, points, points)
    HH, result["hm_assembly"] = pc_measure("HMatrices assembly") do
        return HMatrices.assemble_hmatrix(
            K,
            hcluster,
            hcluster;
            adm,
            comp=HMatrices.PartialACA(; rtol=tol, rank=maxrank),
            threads=true,
            distributed=false,
        )
    end
    merge!(result["hm_assembly"], pc_hmstats(HH))
    if partition !== :native
        identical = pc_blocks(HH) == pc_blocks(H)
        result["identical_partition"] = identical
        println("Identical partitions: ", identical)
    end
    result["hm_assembly"]["probe_error"] =
        norm(HH * probe - reference_probe) / norm(reference_probe)
    result["hm_assembly"]["offdiagonal_probe_error"] =
        result["hm_assembly"]["probe_error"] * norm(reference_probe) /
        offdiagonal_probe_norm
    for sample in 1:samples
        HD, result["hm_input_copy_$sample"] = pc_measure(
            () -> deepcopy(HH), "HMatrices input copy $sample"
        )
        HF, factime = pc_factor(
            () -> lu!(HD; rtol=tol, threads=true), "HMatrices LU $sample"
        )
        if isnothing(HF)
            result["hm_lu_$sample"] = factime
            break
        end
        solution = copy(rhs)
        _, solvetime = pc_measure(() -> ldiv!(HF, solution), "HMatrices solve $sample")
        BLAS.set_num_threads(dense_blas)
        factime["solve_residual"] = norm(A * solution - rhs) / norm(rhs)
        factime["solution_error"] = norm(solution - xtrue) / norm(xtrue)
        factime["dense_solution_error"] = norm(solution - dense_solve) / norm(dense_solve)
        BLAS.set_num_threads(hierarchical_blas)
        merge!(factime, pc_hmstats(HD))
        result["hm_lu_$sample"] = factime
        result["hm_solve_$sample"] = solvetime
        println(
            "HMatrices accuracy: residual=",
            factime["solve_residual"],
            "; solution error=",
            factime["solution_error"],
        )
    end
    println("RESULT ", repr(result))
    flush(stdout)
    return result
end

function pointcloudlu_benchmark(; sizes=(2000, 10000, 50000), output=nothing, kwargs...)
    oldthreads = BLAS.get_num_threads()
    try
        # Compile the same call paths on a genuinely hierarchical small case.
        println("Warmup (excluded from reported cases)")
        pc_case(512; merge((; kwargs...), (; leafsize=16))...)
        results = Dict{String,Any}()
        for n in sizes
            results[string(n)] = pc_case(n; kwargs...)
            if !isnothing(output)
                open(output, "w") do io
                    return TOML.print(io, results)
                end
            end
        end
        return results
    finally
        BLAS.set_num_threads(oldthreads)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    sizes = isempty(ARGS) ? (2000, 10000, 50000) : Tuple(parse.(Int, ARGS))
    pointcloudlu_benchmark(; sizes)
end
