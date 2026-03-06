using AdaptiveCrossApproximation
using LinearAlgebra
using StaticArrays
using Random
using Test

##

pts1 = range(0.0, 1.0; length=100)
pts2 = range(0.0, 1.0; length=110)
K = [exp(-abs2(pj - pk)) for pj in pts1, pk in pts2]

# accuracy validation

for tol in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14]
    local U, V = AdaptiveCrossApproximation.aca(K; tol=tol)
    @test norm(U * V - K) / norm(K) < tol
    local U, V = AdaptiveCrossApproximation.acaᵀ(K; tol=tol)
    @test norm(U * V - K) / norm(K) < tol

    local U, V = AdaptiveCrossApproximation.aca(K; tol=tol, svdrecompress=true)
    @test norm(U * V - K) / norm(K) < tol
    local U, V = AdaptiveCrossApproximation.acaᵀ(K; tol=tol, svdrecompress=true)
    @test norm(U * V - K) / norm(K) < tol
end

# special cases

K = zeros(10, 10)
U, V = AdaptiveCrossApproximation.aca(K; tol=10^-4, maxrank=5)
@test length(U) == 0
@test length(V) == 0

K[4, :] = rand(10)
U, V = AdaptiveCrossApproximation.aca(K; tol=10^-4, maxrank=5)
@test size(U, 2) == 1
@test size(V, 1) == 1

K[1:2, :] = rand(2, 10)
U, V = AdaptiveCrossApproximation.aca(K; tol=10^-4, maxrank=5)
@test size(U, 2) == 3
@test size(V, 1) == 3

# BufferTests
##
comp = AdaptiveCrossApproximation.ACA()
rowbuffer = zeros(Float64, 5, 10)
colbuffer = zeros(Float64, 10, 5)

K = rand(10, 10)
rows = zeros(Int, 5)
cols = zeros(Int, 5)
comp = comp(1:5, 1:5)
npivots = comp(
    K, view(colbuffer, 2:6, 1:5), view(rowbuffer, 1:5, 2:6), rows, cols, Vector(2:6), 2:6, 5
)
@test isapprox(K[2:6, 2:6], colbuffer[2:6, :] * rowbuffer[:, 2:6])

comp = comp(Vector(1:10), Vector(1:10); maxrank=5)
rowstate = comp.rowpivoting.usedidcs
colstate = comp.columnpivoting.usedidcs
comp(Vector(2:6), Vector(2:6); maxrank=5)
@test comp.rowpivoting.usedidcs === rowstate
@test comp.columnpivoting.usedidcs === colstate

npivots_reuse = comp(
    K,
    view(colbuffer, 2:6, 1:5),
    view(rowbuffer, 1:5, 2:6),
    5;
    rows=rows,
    cols=cols,
    rowidcs=Vector(2:6),
    colidcs=Vector(2:6),
)
@test npivots_reuse > 0
##
compT = AdaptiveCrossApproximation.ACAᵀ()
rowsT = zeros(Int, 5)
colsT = zeros(Int, 5)
rowbufferT = zeros(Float64, 5, 10)
colbufferT = zeros(Float64, 10, 5)

npivots = compT(
    transpose(K),
    view(colbufferT, 2:6, 1:5),
    view(rowbufferT, 1:5, 2:6),
    5;
    rows=rowsT,
    cols=colsT,
    rowidcs=2:6,
    colidcs=Vector(2:6),
)

@test isapprox(transpose(K)[2:6, 2:6], colbufferT[2:6, :] * rowbufferT[:, 2:6])
@test rowsT == cols
@test colsT == rows

##

@testset "ACA" begin
    Random.seed!(1)
    tpos = [@SVector rand(3) for _ in 1:48]
    spos = [@SVector rand(3) for _ in 1:52] .+ Scalar(SVector(3.5, 0.0, 0.0))
    Kc = [inv(norm(tp - sp)) for tp in tpos, sp in spos]

    rowpivotings = [
        AdaptiveCrossApproximation.MaximumValue(), AdaptiveCrossApproximation.Leja2(tpos)
    ]
    colpivotings = [
        AdaptiveCrossApproximation.MaximumValue(),
        AdaptiveCrossApproximation.Leja2(spos),
        AdaptiveCrossApproximation.FillDistance(spos),
    ]
    mk_convergence = [
        AdaptiveCrossApproximation.FNormEstimator(1e-4),
        AdaptiveCrossApproximation.FNormExtrapolator(1e-4),
        AdaptiveCrossApproximation.RandomSampling(; tol=1e-4, nsamples=120),
        AdaptiveCrossApproximation.CombinedConvCrit([
            AdaptiveCrossApproximation.FNormEstimator(1e-4),
            AdaptiveCrossApproximation.RandomSampling(; tol=1e-4, nsamples=120),
        ]),
    ]

    for rowpivoting in rowpivotings
        for colpivoting in colpivotings
            for convergence in mk_convergence
                U, V = AdaptiveCrossApproximation.aca(
                    Kc;
                    rowpivoting=rowpivoting,
                    columnpivoting=colpivoting,
                    convergence=convergence,
                    maxrank=30,
                )

                @test size(U, 1) == size(Kc, 1)
                @test size(V, 2) == size(Kc, 2)
                @test size(U, 2) == size(V, 1)
                @test norm(U * V - Kc) / norm(Kc) < 2e-4
            end
        end
    end
end
