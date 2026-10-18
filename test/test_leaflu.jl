using AdaptiveCrossApproximation
using LinearAlgebra
using Random
using Test

function leaf_test_matrix(rng, T, n)
    A = randn(rng, T, n, n) + n * I
    # Force row swaps; n >= 3 gives a non-self-inverse permutation.
    return A[circshift(collect(1:n), 1), :]
end

function leaf_test_allocations(F)
    AdaptiveCrossApproximation.leaflu!(F)
    return @allocated AdaptiveCrossApproximation.leaflu!(F)
end

@testset "Dense leaf LU" begin
    ACA = AdaptiveCrossApproximation
    rng = MersenneTwister(92381)
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        tol = 200eps(real(T))
        @testset "$T" begin
            for n in (0, 1, 3, 8, 31)
                A = leaf_test_matrix(rng, T, n)
                storage = copy(A)
                F = ACA.leaflu!(storage; node=17)
                reference = lu(copy(A))
                @test F isa Factorization{T}
                @test F.factors === storage
                @test F.node == 17
                @test F.info == 0
                @test issuccess(F)
                @test size(F) == size(A)
                @test size(F, 2) == n
                @test F.ipiv == reference.ipiv
                @test F.factors ≈ reference.factors rtol=tol
                L, U = UnitLowerTriangular(F.factors), UpperTriangular(F.factors)
                PA = copy(A)
                ACA._leafpivots!(PA, F.ipiv, false)
                @test PA ≈ L * U rtol=tol
                pivots = F.ipiv
                pivotptr = pointer(pivots)
                copyto!(F.factors, A)
                @test ACA.leaflu!(F) === F
                @test F.ipiv === pivots
                @test pointer(F.ipiv) == pivotptr
                @test F.factors ≈ reference.factors rtol=tol
            end

            @testset "Pivot direction and inversion" begin
                A = T[0 1 0; 0 0 1; 1 0 0]
                F = ACA.leaflu!(copy(A))
                P = Matrix{T}(I, 3, 3)
                ACA._leafpivots!(P, F.ipiv, false)
                @test P != transpose(P)
                @test P * A == I
                for inverse in (false, true)
                    B = randn(rng, T, 3, 3)
                    original = copy(B)
                    ACA._leafpivots!(B, F.ipiv, inverse)
                    @test B == (inverse ? transpose(P) : P) * original
                    ACA._leafpivots!(B, F.ipiv, !inverse)
                    @test B == original
                end
            end

            @testset "Warm allocations" begin
                F = ACA.leaflu!(Matrix{T}(I, 8, 8))
                leaf_test_allocations(F)
                @test leaf_test_allocations(F) == 0
            end

            @testset "Factorization in a padded dense leaf view" begin
                A = leaf_test_matrix(rng, T, 7)
                parent = fill(T(23), 12, 11)
                leaf = view(parent, 3:9, 2:8)
                copyto!(leaf, A)
                F = ACA.leaflu!(leaf)
                @test F.factors === leaf
                @test F.factors ≈ lu(A).factors rtol=tol
                @test all(==(T(23)), view(parent, 1:2, :))
                @test all(==(T(23)), view(parent, :, 1))
            end
        end
    end

    @testset "Failure state and validation" begin
        F = ACA.leaflu!([0.0 1; 0 2]; node=42, check=false)
        @test !issuccess(F)
        @test F.info == 1
        error = try
            ACA.leaflu!([0.0 1; 0 2]; node=42)
        catch e
            e
        end
        @test error isa LeafLUFailure
        @test error.node == 42 && error.pivot == 1
        @test occursin("cluster 42, local pivot 1", sprint(showerror, error))
        @test_throws LeafLUFailure ACA._checkleafstate(F)
        @test_throws DimensionMismatch ACA.leaflu!(zeros(2, 3))
        @test_throws ArgumentError ACA.leaflu!(view(zeros(4, 2), 1:2:4, :))
        @test_throws ArgumentError ACA.leaflu!([NaN 0.0; 0 1])
        @test_throws ArgumentError ACA.leaflu!([1.0 Inf; 0 1])
        F = ACA.leaflu!(Matrix{Float64}(I, 2, 2))
        F.ipiv[1] = 0
        @test_throws ArgumentError ACA._checkleafpivots(F)
        F.factors[1, 1] = NaN
        @test_throws ArgumentError ACA.leaflu!(F)
        @test F.info == -1
        @test !issuccess(F)
        @test_throws ArgumentError ACA._checkleafstate(F)
        copyto!(F.factors, Matrix{Float64}(I, 2, 2))
        @test issuccess(ACA.leaflu!(F))

        # Restricted pivoting can fail even though the complete matrix is invertible.
        A = [0.0 1; 1 0]
        @test isfinite(cond(A))
        @test_throws LeafLUFailure ACA.leaflu!(copy(A[1:1, 1:1]); node=1)
    end
end
