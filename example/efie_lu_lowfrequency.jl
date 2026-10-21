# Timing of the hierarchical LU for the EFIE on a PEC sphere, low frequency:
# λ = 4 × diameter on every mesh.
#
#   julia -t 16 --project=$HOME/Documents/workspace example/efie_lu_lowfrequency.jl

using LinearAlgebra
using CompScienceMeshes
using BEAST
using ParallelKMeans
using H2Trees
using AdaptiveCrossApproximation
using AdaptiveCrossApproximation: CombinedConvCrit, CombinedPivStrat
using AdaptiveCrossApproximation: RandomSampling, RandomSamplingPivoting
using AdaptiveCrossApproximation: LUWorkspace, SolveWorkspace

BLAS.set_num_threads(1)

tol = 1e-3
# As the default for the double-layer operators in the BEAST extension.
compressor = ACA(;
    rowpivoting=CombinedPivStrat([MaximumValue(), RandomSamplingPivoting(1)]),
    convergence=CombinedConvCrit([FNormEstimator(tol), RandomSampling(; tol=tol)]),
)

radius = 1.0
λ = 4 * 2radius
κ = 2π / λ

results = joinpath(@__DIR__, "efie_lu_lowfrequency.txt")
open(results, "w") do io
    return println(io, "N ntasks tassembly tws_lu tlu tws_solve tsolve tassembly_dense tlu_dense tsolve_dense")
end

# Icosphere with subdivision frequency ν: N = 30ν² RWG functions.
for ν in [26, 37, 52, 73]
    Γ = meshicosphere(ν, radius)
    X = raviartthomas(Γ)
    N = numfunctions(X)
    t = Maxwell3D.singlelayer(; wavenumber=κ)
    E = Maxwell3D.planewave(; direction=ẑ, polarization=x̂, wavenumber=κ)
    e = assemble((n × E) × n, X)

    # Dense reference up to 80k unknowns (NaN above). The dense LU runs on 8 BLAS
    # threads, as many as the tasks of the hierarchical LU.
    tassembly_dense = tlu_dense = tsolve_dense = NaN
    if ν <= 52
        tassembly_dense = @elapsed Z = assemble(t, X, X)
        BLAS.set_num_threads(8)
        tlu_dense = @elapsed Fdense = lu!(Z)
        tsolve_dense = @elapsed udense = Fdense \ e
        BLAS.set_num_threads(1)
        @info "dense" N tassembly_dense tlu_dense tsolve_dense
        Z = Fdense = nothing
        GC.gc()
    end

    # One cluster tree on both axes: the LU requires the same tree object.
    tassembly = @elapsed begin
        builder = KMeansTreeBuilder(; numberofclusters=2, minvalues=128)
        cluster = KMeansTree(X; builder=builder)
        tree = BlockTree(cluster, cluster)
        H = HMatrix(
            t,
            X,
            X,
            tree;
            tol=tol,
            maxrank=50,
            compressor=compressor,
            ntasks=16,
            verbose=false,
        )
    end

    # The LU runs on ntasks tasks; the solve is serial. Each LU factorizes a
    # copy, since lu! overwrites its argument.
    for ntasks in [8]
        A = copy(H)
        tws_lu = @elapsed ws = LUWorkspace(A; ntasks=ntasks)
        tlu = @elapsed F = lu!(A, ws)
        tws_solve = @elapsed S = SolveWorkspace(F)
        tsolve = @elapsed u = S \ e

        @info "low frequency" N ntasks tassembly tws_lu tlu tws_solve tsolve tlu_dense
        open(results, "a") do io
            row = (
                N,
                ntasks,
                tassembly,
                tws_lu,
                tlu,
                tws_solve,
                tsolve,
                tassembly_dense,
                tlu_dense,
                tsolve_dense,
            )
            return println(io, join(row, " "))
        end
    end
end
