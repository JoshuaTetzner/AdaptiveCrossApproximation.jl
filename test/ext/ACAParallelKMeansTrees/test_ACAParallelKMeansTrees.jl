using AdaptiveCrossApproximation
using H2Trees
using LinearAlgebra
using Random
using StaticArrays
using Test
using BEAST
using CompScienceMeshes
using ParallelKMeans

@testset "ACAParallelKMeansTrees.jl" begin
    @testset "assemble / defaulttreebackend (ParallelKMeans)" begin
        # ParallelKMeans is loaded above (for the KMeansTree/BlockTree cell earlier in
        # this file), so assemble's default tree backend should now resolve to the
        # k-means clustered tree (ACAParallelKMeansTrees extension) instead of TwoNTree.
        let
            Γkmeans = CompScienceMeshes.meshsphere(1.0, 0.3)
            Xkmeans = raviartthomas(Γkmeans)
            opkmeans = Maxwell3D.singlelayer(; wavenumber=1.0)

            @test AdaptiveCrossApproximation.defaulttreebackend() isa
                AdaptiveCrossApproximation.KMeansTreeBackend

            Tcompressed = AdaptiveCrossApproximation.assemble(
                opkmeans, Xkmeans, Xkmeans; tol=1e-3, maxrank=40
            )
            Tdense = assemble(opkmeans, Xkmeans, Xkmeans)

            @test norm(Matrix(Tcompressed) - Tdense) / norm(Tdense) < 1e-2
        end
    end
end
