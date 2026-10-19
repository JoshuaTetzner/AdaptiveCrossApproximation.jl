using AdaptiveCrossApproximation
using H2Trees
using LinearAlgebra
using Random
using StaticArrays
using Test
using BEAST
using CompScienceMeshes

@testset "assembly.jl" begin
    @testset "assemble / defaulttreebackend (H2Trees)" begin
        Γhassemble = CompScienceMeshes.meshsphere(1.0, 0.3)
        Xhassemble = raviartthomas(Γhassemble)
        ophassemble = Maxwell3D.singlelayer(; wavenumber=1.0)

        @test AdaptiveCrossApproximation.defaulttreebackend() isa
            AdaptiveCrossApproximation.H2Tree

        Tcompressed = AdaptiveCrossApproximation.assemble(
            ophassemble, Xhassemble, Xhassemble; tol=1e-3, maxrank=40
        )
        Tdense = assemble(ophassemble, Xhassemble, Xhassemble)

        @test norm(Matrix(Tcompressed) - Tdense) / norm(Tdense) < 1e-2
    end
end
