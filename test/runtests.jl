using Test, TestItems, TestItemRunner
using BEAST
using CompScienceMeshes
using H2Trees
using LinearAlgebra
using Random
using StaticArrays
using AdaptiveCrossApproximation

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
    Aqua.test_all(AdaptiveCrossApproximation; deps_compat=false)
end

@testitem "Code formatting (JuliaFormatter.jl)" begin
    using JuliaFormatter
    pkgpath = pkgdir(AdaptiveCrossApproximation)
    @test JuliaFormatter.format(pkgpath, overwrite=true)
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

@testitem "AdaptiveCrossApproximation" begin
    include("test_pivoting.jl")
    include("test_orientations.jl")
    include("test_convergence.jl")
    include("test_kernelmatrix.jl")
    include("test_utils.jl")

    include("test_aca.jl")
    include("test_acaT.jl")
    include("test_iaca.jl")
    include("test_acabeast.jl")

    include("test_hmatrix.jl")
end

@run_package_tests verbose = true
