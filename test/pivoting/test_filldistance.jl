using AdaptiveCrossApproximation
using LinearAlgebra
using Random
using StaticArrays
using Test

@testset "filldistance.jl" begin
    @testset "FillDistance" begin
        pts = [@SVector [Float64(i), 0.0, 0.0] for i in 1:8]
        piv = AdaptiveCrossApproximation.FillDistance(pts)
        idcs = [2, 4, 6, 8]
        functor = piv(idcs)

        first = functor()
        @test first == 1

        h_before = copy(functor.h)
        local_idx = functor(zeros(length(idcs)))
        global_idx = idcs[local_idx]

        expected = similar(h_before)
        for i in eachindex(h_before)
            expected[i] = min(h_before[i], norm(pts[idcs[i]] - pts[global_idx]))
        end

        @test all(isapprox.(functor.h, expected))

        resize!(functor, 6)
        @test length(functor.h) == 6
        @test length(functor.idcs) == 6

        reset!(functor, [1, 2, 3, 4, 5, 6])
        @test collect(view(functor.idcs, 1:6)) == [1, 2, 3, 4, 5, 6]
        @test all(iszero, view(functor.h, 1:6))

        pts_fd = [
            SVector(0.0, 0.0, 0.0),
            SVector(0.7, 0.0, 0.0),
            SVector(1.6, 0.0, 0.0),
            SVector(2.4, 0.0, 0.0),
            SVector(3.1, 0.0, 0.0),
            SVector(5.0, 0.0, 0.0),
            SVector(8.0, 0.0, 0.0),
        ]
        idcs_fd = collect(1:length(pts_fd))
        piv_fd = AdaptiveCrossApproximation.FillDistance(pts_fd)
        functor_fd = piv_fd(idcs_fd)

        @test functor_fd() == 1

        for _ in 1:4
            h_before_fd = copy(functor_fd.h)
            objs = similar(h_before_fd)

            for k in eachindex(idcs_fd)
                candidate_global = idcs_fd[k]
                objs[k] = maximum(
                    min(
                        h_before_fd[i], norm(pts_fd[idcs_fd[i]] - pts_fd[candidate_global])
                    ) for i in eachindex(idcs_fd)
                )
            end

            best = minimum(objs)
            minimizers = findall(v -> isapprox(v, best; atol=1e-12, rtol=1e-12), objs)

            chosen = functor_fd(zeros(length(idcs_fd)))
            @test chosen in minimizers
        end

        resize!(functor_fd, 9)
        @test length(functor_fd.h) == 9
        @test length(functor_fd.idcs) == 9

        reset!(functor_fd, collect(1:9))
        @test collect(view(functor_fd.idcs, 1:9)) == collect(1:9)
        @test all(iszero, view(functor_fd.h, 1:9))

        resize!(functor_fd, 12)
        reset!(functor_fd, [2, 4, 6])
        @test length(functor_fd.h) == 12
        @test length(functor_fd.idcs) == 12
        @test functor_fd.nactive == 3
        @test functor_fd(zeros(3)) == 1
        @test all(isapprox.(view(functor_fd.h, 1:3), [0.0, 1.7, 4.3]))
    end
end
