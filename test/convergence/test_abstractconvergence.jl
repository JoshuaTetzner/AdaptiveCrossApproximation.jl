using AdaptiveCrossApproximation
using LinearAlgebra
using Random
using Test

@testset "abstractconvergence.jl" begin
    @testset "_zeroresidualtol (vector residuals)" begin
        for T in (Float32, Float64), F in (Float32, Float64), scale in (1e-16, 1.0, 1e16)
            tol = T == Float32 ? 1e-5 : 1e-12
            relative = T == Float32 ? 5e-5 : 5e-10
            for extrapolate in (false, true)
                criterion = FNormEstimator(F(tol))
                functor = extrapolate ? FNormExtrapolator(criterion)(4) : criterion()
                estimator = extrapolate ? functor.estimator : functor
                estimator.normUV = F(scale)
                # Meaningful residuals below sqrt(eps) must still count as pivots.
                residual = Complex{T}[T(scale*relative)]
                npivot, keepgoing = functor(residual, 2)
                @test npivot == 2
                @test keepgoing
                # A few ulps relative to the block norm remain numerical zero.
                roundoff = Complex{T}[T(scale) * T(4eps(T))]
                npivot, keepgoing = functor(roundoff, 2)
                @test npivot == 1
                @test !keepgoing
            end
        end
    end

    @testset "_zeroresidualtol (matrix residuals)" begin
        for T in (Float32, Float64), scale in (1e-16, 1.0, 1e16)
            tol = T == Float32 ? 1e-5 : 1e-12
            relative = T == Float32 ? 5e-5 : 5e-10
            functor = FNormEstimator(tol)()
            functor.normUV = scale
            # ACA scales the row by its pivot; the column carries the matrix scale.
            rowbuffer = zeros(T, 2, 3)
            colbuffer = zeros(T, 4, 2)
            rowbuffer[2, :] .= (1, 0.5, 0)
            colbuffer[:, 2] .= T(scale * relative)
            npivot, keepgoing = functor(rowbuffer, colbuffer, 2, 4, 3)
            @test npivot == 2
            @test keepgoing
            functor.normUV = scale
            colbuffer[:, 2] .= T(scale) * T(2eps(T))
            npivot, keepgoing = functor(rowbuffer, colbuffer, 2, 4, 3)
            @test npivot == 1
            @test !keepgoing
        end
    end
end
