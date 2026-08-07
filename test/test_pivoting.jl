using AdaptiveCrossApproximation
using LinearAlgebra
using Random
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

@testset "TreeMimicryPivoting" begin
    struct MockTree
        centers::Vector{SVector{3,Float64}}
        nodevalues::Vector{Vector{Int}}
        nodechildren::Vector{Vector{Int}}
        nodeparent::Vector{Int}
    end

    AdaptiveCrossApproximation.center(tree::MockTree, node::Int) = tree.centers[node]
    AdaptiveCrossApproximation.values(tree::MockTree, node::Int) = tree.nodevalues[node]
    AdaptiveCrossApproximation.values(tree::MockTree, nodes::Vector{Int}) =
        reduce(vcat, (tree.nodevalues[n] for n in nodes); init=Int[])
    AdaptiveCrossApproximation.children(tree::MockTree, node::Int) = tree.nodechildren[node]
    AdaptiveCrossApproximation.parent(tree::MockTree, node::Int) = tree.nodeparent[node]
    function AdaptiveCrossApproximation.firstchild(tree::MockTree, node::Int)
        if isempty(tree.nodechildren[node])
            return 0
        else
            return first(tree.nodechildren[node])
        end
    end

    tree = MockTree(
        [
            SVector(4.5, 0.0, 0.0),
            SVector(2.5, 0.0, 0.0),
            SVector(6.5, 0.0, 0.0),
            SVector(1.5, 0.0, 0.0),
            SVector(3.5, 0.0, 0.0),
            SVector(5.5, 0.0, 0.0),
            SVector(7.5, 0.0, 0.0),
        ],
        [collect(1:8), collect(1:4), collect(5:8), [1, 2], [3, 4], [5, 6], [7, 8]],
        [[2, 3], [4, 5], [6, 7], Int[], Int[], Int[], Int[]],
        [0, 1, 1, 2, 2, 3, 3],
    )

    refpos = [@SVector [Float64(i), 0.0, 0.0] for i in 1:8]
    pos = [@SVector [Float64(i), 1.0, 0.0] for i in 1:8]
    piv = AdaptiveCrossApproximation.TreeMimicryPivoting(refpos, pos, tree)
    functor = piv([1, 8], [2, 3], 5)

    @test length(functor.usedidcs) == 5
    @test length(functor.emptyclusters) == 5
    @test functor.nactive == 2
    @test length(functor.h) == 2
    @test length(functor.leja) == 2
    @test length(functor.w) == 2
    @test collect(view(functor.farfield, 1:2)) == [2, 3]

    firstpivot = functor()
    @test firstpivot in 1:8
    secondpivot = functor(2)
    @test secondpivot in 1:8

    usedbuf = functor.usedidcs
    emptybuf = functor.emptyclusters
    farfieldbuf = functor.farfield
    hbuf = functor.h
    lejabuf = functor.leja
    wbuf = functor.w

    resize!(functor, 9)
    @test length(functor.farfield) == 9
    @test length(functor.h) == 9
    @test length(functor.leja) == 9
    @test length(functor.w) == 9
    @test length(functor.usedidcs) == 5
    @test length(functor.emptyclusters) == 5
    @test functor.nactive == 9

    resize!(functor, 4)
    @test length(functor.farfield) == 9
    @test length(functor.h) == 9
    @test length(functor.leja) == 9
    @test length(functor.w) == 9
    @test functor.nactive == 4

    reset!(functor, [1, 8], [2, 3])
    @test functor.usedidcs === usedbuf
    @test functor.emptyclusters === emptybuf
    @test functor.farfield === farfieldbuf
    @test functor.h === hbuf
    @test functor.leja === lejabuf
    @test functor.w === wbuf
    @test functor.nempty == 0
    @test functor.nactive == 2
    @test collect(view(functor.farfield, 1:2)) == [2, 3]
    @test all(iszero, view(functor.h, 1:2))
    @test all(isone, view(functor.leja, 1:2))
    @test all(iszero, view(functor.w, 1:2))
    @test all(iszero, view(functor.usedidcs, 1:5))
    @test all(iszero, view(functor.emptyclusters, 1:5))

    reset!(functor, [1, 8], [2, 3, 4])
    @test functor.nactive == 3
    @test collect(view(functor.farfield, 1:3)) == [2, 3, 4]
    @test length(functor.farfield) == 9
    @test length(functor.h) == 9
    @test length(functor.leja) == 9
    @test length(functor.w) == 9
    @test length(functor.usedidcs) == 5
    @test length(functor.emptyclusters) == 5
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

    reset!(functor_fd, [2, 4, 6, 1, 5, 6, 2])
    @test length(functor_fd.strats[1].usedidcs) == 7
    @test length(functor_fd.strats[2].h) == 7
    @test functor_fd.strats[2].nactive == 7

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

@testset "RandomSamplingPivoting" begin
    K = reshape(collect(1.0:12.0), 3, 4)
    rowidcs = [1, 2, 3]
    colidcs = [1, 2, 3, 4]

    Random.seed!(11)
    cc = AdaptiveCrossApproximation.RandomSampling(; nsamples=5, tol=0.2)
    convcrit = cc(K, rowidcs, colidcs)

    rowpiv = AdaptiveCrossApproximation.RandomSamplingPivoting(1)(convcrit)
    colpiv = AdaptiveCrossApproximation.RandomSamplingPivoting(2)(convcrit)
    @test rowpiv.rc == 1
    @test colpiv.rc == 2
    @test resize!(rowpiv) === nothing
    @test reset!(rowpiv) === nothing

    rowbuffer = zeros(2, 4)
    colbuffer = zeros(3, 2)
    rowbuffer[1, :] .= [1.0, 2.0, 0.0, 0.0]
    colbuffer[:, 1] .= [1.0, 0.0, 1.0]
    convcrit(rowbuffer, colbuffer, 1, 3, 4)

    expected = convcrit.indices[argmax(abs.(view(convcrit.rest, 1:(convcrit.nactive))))]
    @test rowpiv(Float64[]) == expected[1]
    @test colpiv(Float64[]) == expected[2]

    combined = AdaptiveCrossApproximation.CombinedConvCritFunctor(
        AdaptiveCrossApproximation.ConvCritFunctor[convcrit], [true]
    )
    piv2 = AdaptiveCrossApproximation.RandomSamplingPivoting(1)(combined)
    @test piv2(Float64[]) == expected[1]

    badcombined = AdaptiveCrossApproximation.CombinedConvCritFunctor(
        AdaptiveCrossApproximation.ConvCritFunctor[AdaptiveCrossApproximation.FNormEstimator(
            1e-4
        )()],
        [true],
    )
    @test_throws ArgumentError AdaptiveCrossApproximation.RandomSamplingPivoting(1)(
        badcombined
    )
end

@testset "TreeMimicryPivoting2 (DirectionFilter)" begin
    struct DirTree
        nodes::Vector{Int}
        centers::Vector{SVector{3,Float64}}
        nodevalues::Vector{Vector{Int}}
        nodechildren::Vector{Vector{Int}}
        nodeparent::Vector{Int}
    end

    AdaptiveCrossApproximation.center(tree::DirTree, node::Int) = tree.centers[node]
    AdaptiveCrossApproximation.values(tree::DirTree, node::Int) = tree.nodevalues[node]
    AdaptiveCrossApproximation.values(tree::DirTree, nodes::Vector{Int}) =
        reduce(vcat, (tree.nodevalues[n] for n in nodes); init=Int[])
    AdaptiveCrossApproximation.children(tree::DirTree, node::Int) = tree.nodechildren[node]
    AdaptiveCrossApproximation.parent(tree::DirTree, node::Int) = tree.nodeparent[node]
    function AdaptiveCrossApproximation.firstchild(tree::DirTree, node::Int)
        isempty(tree.nodechildren[node]) && return 0
        return first(tree.nodechildren[node])
    end

    # left half (nodes 2,4,5 / basis 1:4) carries direction 1 (ex), right half
    # (nodes 3,6,7 / basis 5:8) carries direction 2 (ey); root sees both.
    tree = DirTree(
        collect(1:7),
        [
            SVector(4.5, 0.0, 0.0),
            SVector(2.5, 0.0, 0.0),
            SVector(6.5, 0.0, 0.0),
            SVector(1.5, 0.0, 0.0),
            SVector(3.5, 0.0, 0.0),
            SVector(5.5, 0.0, 0.0),
            SVector(7.5, 0.0, 0.0),
        ],
        [collect(1:8), collect(1:4), collect(5:8), [1, 2], [3, 4], [5, 6], [7, 8]],
        [[2, 3], [4, 5], [6, 7], Int[], Int[], Int[], Int[]],
        [0, 1, 1, 2, 2, 3, 3],
    )

    edges = fill(SVector(1.0, 0.0, 0.0), 8)
    normals = vcat(fill(SVector(1.0, 0.0, 0.0), 4), fill(SVector(0.0, 1.0, 0.0), 4))
    edgeids, normalids, nedgeids, nnormalids = AdaptiveCrossApproximation.basisfunction_orientation_ids(
        edges, normals
    )
    nodesets, trialnormalids, ntrialnormalids = AdaptiveCrossApproximation.node_normal_orientation_sets(
        normals, tree
    )
    @test normalids == trialnormalids

    refpos = [@SVector [Float64(i), 0.0, 0.0] for i in 1:8]
    pos = [@SVector [Float64(i), 1.0, 0.0] for i in 1:8]

    piv = AdaptiveCrossApproximation.TreeMimicryPivoting2(
        refpos, pos, edgeids, normalids, nodesets, tree
    )
    maxrank = 5
    convcrit = AdaptiveCrossApproximation.PhaseExtrapolator(1e-6)(maxrank)
    functor = piv(convcrit, [1, 8], [2, 3], maxrank)

    # with lastconverged never set (no residual data fed back), the direction
    # filter round-robins between the two available directions every pivot
    used = Int[functor()]
    dirs = Int32[normalids[used[1]]]
    for n in 2:4
        p = functor(n)
        push!(used, p)
        push!(dirs, normalids[p])
    end

    @test length(unique(used)) == 4
    @test dirs == Int32[1, 2, 1, 2]
    @test all(in(1:4), used[1:2:end])
    @test all(in(5:8), used[2:2:end])
end
