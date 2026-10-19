using AdaptiveCrossApproximation
using LinearAlgebra
using Random
using Test

@testset "randomsampling.jl" begin
    @testset "RandomSamplingFunctor" begin
        K = reshape(collect(1.0:12.0), 3, 4)
        rowidcs = [1, 2, 3]
        colidcs = [1, 2, 3, 4]

        Random.seed!(11)
        cc = AdaptiveCrossApproximation.RandomSampling(; nsamples=5, tol=0.2)
        functor = cc(K, rowidcs, colidcs)

        @test length(functor.indices) == 5
        @test length(unique(functor.indices)) == 5
        @test all(
            1 <= rc[1] <= length(rowidcs) && 1 <= rc[2] <= length(colidcs) for
            rc in functor.indices
        )

        expected_rest = [K[rowidcs[rc[1]], colidcs[rc[2]]] for rc in functor.indices]
        @test all(isapprox.(functor.rest, expected_rest))

        rowbuffer = zeros(2, 4)
        colbuffer = zeros(3, 2)
        rowbuffer[1, :] .= [1.0, 2.0, 0.0, 0.0]
        colbuffer[:, 1] .= [1.0, 0.0, 1.0]

        rest_before = copy(functor.rest)
        expected_after = similar(rest_before)
        for i in eachindex(rest_before)
            rc = functor.indices[i]
            expected_after[i] = rest_before[i] - colbuffer[rc[1], 1] * rowbuffer[1, rc[2]]
        end

        rnorm = norm(rowbuffer[1, 1:4])
        cnorm = norm(colbuffer[1:3, 1])
        lhs = sqrt(sum(abs2, expected_after) / length(expected_after) * 3 * 4)
        rhs = functor.convergence.tol * (rnorm * cnorm)

        npivot, conv = functor(rowbuffer, colbuffer, 1, 3, 4)
        @test npivot == 1
        @test conv == (lhs > rhs)
        @test all(isapprox.(functor.rest, expected_after))
        @test isapprox(functor.normUV, rnorm * cnorm)

        rowidcs2 = [1, 3]
        colidcs2 = [2, 4]
        Random.seed!(22)
        reset!(functor, rowidcs2, colidcs2)
        @test iszero(functor.normUV)
        @test length(functor.indices) >= 4
        @test length(functor.rest) >= 4
        @test functor.nactive == 4
        @test length(unique(view(functor.indices, 1:(functor.nactive)))) == 4
        @test all(
            1 <= rc[1] <= length(rowidcs2) && 1 <= rc[2] <= length(colidcs2) for
            rc in view(functor.indices, 1:(functor.nactive))
        )
        expected_rest2 = [
            K[rowidcs2[rc[1]], colidcs2[rc[2]]] for
            rc in view(functor.indices, 1:(functor.nactive))
        ]
        @test all(isapprox.(view(functor.rest, 1:(functor.nactive)), expected_rest2))
    end
end
