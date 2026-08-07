using Test, TestItems, TestItemRunner
using BEAST
using CompScienceMeshes
using H2Trees
using LinearAlgebra
using Random
using StaticArrays
using AdaptiveCrossApproximation

# JET.test_package is intentionally NOT wired in as a @testitem: JET.jl's analyzer
# throws an internal "Expected MethodTableView" error whenever it runs inside
# TestItemRunner's execution model (each @testitem is `Base.invokelatest`-eval'd
# into a fresh anonymous module), regardless of ordering relative to other
# testitems — confirmed reproducible with JET running first, last, and
# immediately after/before Aqua.test_all. This is a JET.jl / TestItemRunner.jl
# incompatibility, not a defect in this package: calling
# `JET.report_package(AdaptiveCrossApproximation; target_modules=(AdaptiveCrossApproximation,))`
# directly (outside the test harness, with BEAST/CUDA/H2Trees all loaded, both
# before and after Aqua.test_all) reports zero errors. Run it manually instead:
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
