using Test, TestItems, TestItemRunner
using AdaptiveCrossApproximation

@testitem "AdaptiveCrossApproximation" begin
    include("test_aca.jl")
    include("test_convergence.jl")
    include("test_pivoting.jl")
end

@testitem "Code quality (Aqua.jl)" begin
    using Aqua
    Aqua.test_all(AdaptiveCrossApproximation; deps_compat=false)
end
@testitem "Code linting (JET.jl)" begin
    using JET
    JET.test_package(AdaptiveCrossApproximation; target_defined_modules=true)
end

@testitem "Code formatting (JuliaFormatter.jl)" begin
    using JuliaFormatter
    pkgpath = pkgdir(AdaptiveCrossApproximation)
    @test JuliaFormatter.format(pkgpath, overwrite=false)
end

@run_package_tests verbose = true
