using AdaptiveCrossApproximation
using LinearAlgebra
using StaticArrays
using Test

@testset "Leja2" begin
    pts = [@SVector [Float64(i), 0.0, 0.0] for i in 1:6]
    piv = AdaptiveCrossApproximation.Leja2(pts)
    idcs = [2, 4, 6]
    functor = piv(idcs)

    @test functor() == 1
    @test all(isapprox.(functor.h, [0.0, 2.0, 4.0]))

    next = functor(zeros(length(idcs)))
    @test next == 3
    @test all(isapprox.(functor.h, [0.0, 2.0, 0.0]))

    resize!(functor, 5)
    @test length(functor.h) == 5
    @test length(functor.idcs) == 5

    reset!(functor, [1, 3, 5, 6, 2])
    @test collect(view(functor.idcs, 1:5)) == [1, 3, 5, 6, 2]
    @test all(iszero, view(functor.h, 1:5))

    resize!(functor, 8)
    reset!(functor, [1, 3, 5])
    @test length(functor.h) == 8
    @test length(functor.idcs) == 8
    @test functor.nactive == 3
    @test functor(zeros(3)) == 1
    @test all(isapprox.(view(functor.h, 1:3), [0.0, 2.0, 4.0]))
end

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
                min(h_before_fd[i], norm(pts_fd[idcs_fd[i]] - pts_fd[candidate_global])) for
                i in eachindex(idcs_fd)
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

@testset "CombinedPivStrat" begin
    pts = [SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0), SVector(2.0, 0.0, 0.0)]
    strats = [
        AdaptiveCrossApproximation.MaximumValue(), AdaptiveCrossApproximation.Leja2(pts)
    ]
    comb = AdaptiveCrossApproximation.CombinedPivStrat(strats)

    conv = AdaptiveCrossApproximation.CombinedConvCritFunctor(
        [
            AdaptiveCrossApproximation.FNormEstimator(1e-4)(),
            AdaptiveCrossApproximation.FNormEstimator(1e-4)(),
        ],
        [true, false],
    )

    functor = comb(conv, [1, 2, 3])

    idx1 = functor([0.1, 2.0, 0.2])
    @test idx1 == 2

    conv.isconverged .= [false, true]
    idx2 = functor([0.1, 2.0, 0.2])
    @test idx2 == 1

    resize!(functor, 5)
    @test length(functor.strats[1].usedidcs) == 5
    @test length(functor.strats[2].idcs) == 5
    @test length(functor.strats[2].h) == 5

    reset!(functor, [3, 1, 2, 3, 1])
    @test all(.!view(functor.strats[1].usedidcs, 1:5))
    @test collect(view(functor.strats[2].idcs, 1:5)) == [3, 1, 2, 3, 1]
    @test all(iszero, view(functor.strats[2].h, 1:5))

    pts_fd = [@SVector [Float64(i), 0.0, 0.0] for i in 1:6]
    strats_fd = [
        AdaptiveCrossApproximation.MaximumValue(),
        AdaptiveCrossApproximation.FillDistance(pts_fd),
    ]
    comb_fd = AdaptiveCrossApproximation.CombinedPivStrat(strats_fd)

    conv_fd = AdaptiveCrossApproximation.CombinedConvCritFunctor(
        [
            AdaptiveCrossApproximation.FNormEstimator(1e-4)(),
            AdaptiveCrossApproximation.FNormEstimator(1e-4)(),
        ],
        [false, true],
    )

    idcs_fd = [2, 4, 6]
    functor_fd = comb_fd(conv_fd, idcs_fd)
    chosen_fd = functor_fd(zeros(length(idcs_fd)))

    @test chosen_fd == 1
    @test all(isapprox.(functor_fd.strats[2].h, [0.0, 2.0, 4.0]))

    resize!(functor_fd, 7)
    reset!(functor_fd, [2, 4, 6])
    chosen_fd_reset = functor_fd(zeros(3))
    @test chosen_fd_reset == 1
    @test length(functor_fd.strats[2].h) == 7
    @test functor_fd.strats[2].nactive == 3
    @test all(isapprox.(view(functor_fd.strats[2].h, 1:3), [0.0, 2.0, 4.0]))
end

@testset "MaximumValue" begin
    piv = AdaptiveCrossApproximation.MaximumValue()
    idcs = [10, 20, 30, 40]
    functor = piv(idcs)

    @test functor() == 1
    next = functor([0.2, -3.0, 1.0, 2.5])
    @test next == 2

    AdaptiveCrossApproximation.reset!(functor, [1, 2, 3])
    @test length(functor.usedidcs) >= 3
    @test all(.!view(functor.usedidcs, 1:3))

    resize!(functor, 6)
    @test length(functor.usedidcs) == 6

    AdaptiveCrossApproximation.reset!(functor, [9, 8, 7, 6, 5, 4])
    @test all(.!view(functor.usedidcs, 1:6))

    resize!(functor, 9)
    AdaptiveCrossApproximation.reset!(functor, [1, 2, 3])
    @test length(functor.usedidcs) == 9
    @test functor.nactive == 3
    @test functor([0.1, -2.0, 1.5]) == 2
end

@testset "MimicryPivoting" begin
    refpos = [@SVector [Float64(i), 0.0, 0.0] for i in 1:6]
    pos = [@SVector [Float64(i), 1.0, 0.0] for i in 1:8]
    piv = AdaptiveCrossApproximation.MimicryPivoting(refpos, pos)
    functor = piv([1, 2, 3], [2, 4, 6, 8])

    idbuf = functor.idcs
    hbuf = functor.h
    lejabuf = functor.leja
    wbuf = functor.w

    first = functor()
    @test first in (2, 4, 6, 8)

    reset!(functor, [2, 3], [1, 3, 5])
    @test functor.idcs === idbuf
    @test functor.h === hbuf
    @test functor.leja === lejabuf
    @test functor.w === wbuf
    @test functor.nactive == 3
    @test collect(view(functor.idcs, 1:3)) == [1, 3, 5]

    resize!(functor, 10)
    @test length(functor.idcs) == 10
    @test length(functor.h) == 10
    @test length(functor.leja) == 10
    @test length(functor.w) == 10
    @test functor.nactive == 10

    reset!(functor, [1, 4, 6], [2, 5, 8, 7])
    @test functor.nactive == 4
    @test collect(view(functor.idcs, 1:4)) == [2, 5, 8, 7]
    @test all(iszero, view(functor.h, 1:4))
    @test all(isone, view(functor.leja, 1:4))
end
