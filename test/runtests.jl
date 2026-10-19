using Test, TestItems, TestItemRunner
using BEAST
using CompScienceMeshes
using H2Trees
using LinearAlgebra
using Random
using StaticArrays
using AdaptiveCrossApproximation

Random.seed!(1234)

@testset "AdaptiveCrossApproximation" begin
    # Shared fixtures first; suite files never include one another.
    include("testutils.jl")

    # Follow source-file names and directories. Interface-only files without
    # separate assertions are exercised through their concrete implementations.
    include("test_utils.jl")
    include("hmatrix/kernelmatrix/test_abstractkernelmatrix.jl")
    include("pivoting/test_maxvalue.jl")
    include("pivoting/test_lejapoints.jl")
    include("pivoting/test_filldistance.jl")
    include("pivoting/test_mimicrypivoting.jl")
    include("pivoting/test_treemimicrypivoting.jl")
    include("pivoting/filters/test_orientations.jl")
    include("convergence/test_abstractconvergence.jl")
    include("convergence/test_estimation.jl")
    include("convergence/test_extrapolation.jl")
    include("convergence/test_randomsampling.jl")
    include("convergence/test_combinedconvcrit.jl")
    include("pivoting/filters/test_efiedirectional.jl")
    include("pivoting/test_combinedpivstrat.jl")
    include("pivoting/test_randomsampling.jl")
    include("test_aca.jl")
    include("test_acaT.jl")
    include("test_iaca.jl")
    include("test_fullpivotedaca.jl")

    # hmatrix/abstracthmatrix.jl and its source-file includes.
    include("hmatrix/test_hmatrix.jl")
    include("hmatrix/test_abstracthmatrix.jl")
    include("hmatrix/test_treeindex.jl")

    # Test the H2Trees default before loading ParallelKMeans below.
    include("hmatrix/test_assembly.jl")

    # Optional backends/extensions use the same per-source-file layout.
    include("ext/ACAParallelKMeansTrees/test_ACAParallelKMeansTrees.jl")
    include("hmatrix/arithmetic/test_matvec.jl")
    include("hmatrix/arithmetic/test_taskgraph.jl")
    include("hmatrix/arithmetic/test_recompression.jl")
    include("hmatrix/arithmetic/test_workspace.jl")
    include("hmatrix/arithmetic/test_blockview.jl")
    include("hmatrix/arithmetic/test_hierarchicaladdition.jl")
    include("hmatrix/arithmetic/test_thinmultiplication.jl")
    include("hmatrix/arithmetic/test_hierarchicalmultiplication.jl")
    include("hmatrix/arithmetic/test_leaflu.jl")
    include("hmatrix/arithmetic/test_triangular.jl")
    include("hmatrix/arithmetic/test_solveplan.jl")
    include("hmatrix/arithmetic/test_lu.jl")
    include("ext/ACAH2Trees/test_interactions.jl")
    include("ext/ACABEAST/test_kernelmatrix.jl")
    include("ext/ACACUDA/test_batchedfullpivotaca.jl")
end

# JET.test_package is intentionally NOT wired in as a @testitem: run alongside
# Aqua.test_all with BEAST/CUDA/CompScienceMeshes/H2Trees loaded, it never
# finishes (still running, no output, after 35+ minutes vs. under 30s standalone).
# Run it manually instead:
#
#   using JET, AdaptiveCrossApproximation
#   JET.report_package(AdaptiveCrossApproximation; target_modules=(AdaptiveCrossApproximation,))
#
#@testitem "Static analysis (JET.jl)" begin
#    using JET
#    using AdaptiveCrossApproximation
#    JET.test_package(
#        AdaptiveCrossApproximation; target_modules=(AdaptiveCrossApproximation,)
#    )
#end

@testitem "Code quality (Aqua.jl)" begin
    using Aqua
    Aqua.test_all(AdaptiveCrossApproximation)
end

@testitem "Code formatting (JuliaFormatter.jl)" begin
    using JuliaFormatter
    pkgpath = pkgdir(AdaptiveCrossApproximation)
    @test JuliaFormatter.format(pkgpath, overwrite=false)
end

@testitem "Explicit imports (ExplicitImports.jl)" begin
    using ExplicitImports
    using AdaptiveCrossApproximation
    @test ExplicitImports.check_no_stale_explicit_imports(AdaptiveCrossApproximation) ===
        nothing
    @test ExplicitImports.check_all_explicit_imports_via_owners(
        AdaptiveCrossApproximation
    ) === nothing
    @test ExplicitImports.check_no_self_qualified_accesses(AdaptiveCrossApproximation) ===
        nothing
end

@run_package_tests verbose = true
