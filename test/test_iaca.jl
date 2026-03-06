using AdaptiveCrossApproximation
using LinearAlgebra
using StaticArrays
using Random
using Test
##
Random.seed!(1)
pts1 = [@SVector rand(3) for i in 1:100]
pts2 = [@SVector rand(3) for i in 1:110] .+ Scalar(SVector(4.0, 0.0, 0.0))
K = [1 / (norm(pj - pk)) for pj in pts1, pk in pts2]

iaca = AdaptiveCrossApproximation.iACA(
    AdaptiveCrossApproximation.MimicryPivoting(pts2, pts1),
    AdaptiveCrossApproximation.MaximumValue(),
    AdaptiveCrossApproximation.FNormExtrapolator(
        AdaptiveCrossApproximation.iFNormEstimator(1e-4)
    ),
)
iaca = iaca(Vector(1:100), Vector(1:110); maxrank=40)
colbuffer = zeros(Float64, 40, 40)
rowbuffer = zeros(Float64, 40, 110)
npiv, rows, cols = iaca(K, colbuffer, rowbuffer, 40)

@test norm(K[:, cols] * inv(K[rows, cols]) * K[rows, :] - K) / norm(K) < 1e-4

rowstate = iaca.rowpivoting.idcs
colstate = iaca.columnpivoting.usedidcs
iaca(Vector(1:80), Vector(1:90); maxrank=40)
@test iaca.rowpivoting.idcs === rowstate
@test iaca.columnpivoting.usedidcs === colstate

npiv_reuse, rows_reuse, cols_reuse = iaca(
    K, colbuffer, rowbuffer, 40; rowidcs=Vector(1:80), colidcs=Vector(1:90)
)
@test npiv_reuse > 0
@test length(rows_reuse) == npiv_reuse
@test length(cols_reuse) == npiv_reuse

##
iaca = AdaptiveCrossApproximation.iACA(pts2, pts1)
iaca(Vector(1:110), Vector(1:100))

colbuffer = zeros(Float64, 110, 40)
rowbuffer = zeros(Float64, 40, 40)
npivT, rowsT, colsT = iaca(Matrix(transpose(K)), colbuffer, rowbuffer, 40)

K = transpose(K)
@test norm(K[:, colsT] * inv(K[rowsT, colsT]) * K[rowsT, :] - K) / norm(K) < 1e-4

@test npiv == npivT
@test rows == colsT
@test cols == rowsT

@testset "iACA contract: supported combinations" begin
    Random.seed!(7)
    nrow = 50
    ncol = 55
    rowpos = [@SVector rand(3) for _ in 1:nrow]
    colpos = [@SVector rand(3) for _ in 1:ncol] .+ Scalar(SVector(3.5, 0.0, 0.0))
    Kc = [inv(norm(rp - cp)) for rp in rowpos, cp in colpos]

    struct LeafTree{D,F}
        centers::Vector{SVector{D,F}}
    end
    AdaptiveCrossApproximation.center(tree::LeafTree, node::Int) = tree.centers[node]
    AdaptiveCrossApproximation.values(::LeafTree, node::Int) = [node]
    AdaptiveCrossApproximation.children(::LeafTree, ::Int) = Int[]
    AdaptiveCrossApproximation.parent(::LeafTree, node::Int) = node
    AdaptiveCrossApproximation.firstchild(::LeafTree, ::Int) = 0

    rowtree = LeafTree(rowpos)
    coltree = LeafTree(colpos)
    conv() = AdaptiveCrossApproximation.FNormExtrapolator(
        AdaptiveCrossApproximation.iFNormEstimator(5e-4)
    )

    iaca_row_mimic = AdaptiveCrossApproximation.iACA(
        AdaptiveCrossApproximation.MimicryPivoting(colpos, rowpos),
        AdaptiveCrossApproximation.MaximumValue(),
        conv(),
    )
    colbuffer = zeros(Float64, 30, 30)
    rowbuffer = zeros(Float64, 30, ncol)
    npiv1, rows1, cols1 = iaca_row_mimic(Kc, colbuffer, rowbuffer, 30)
    @test npiv1 > 0
    @test norm(Kc[:, cols1] * inv(Kc[rows1, cols1]) * Kc[rows1, :] - Kc) / norm(Kc) < 8e-2

    iaca_col_mimic = AdaptiveCrossApproximation.iACA(
        AdaptiveCrossApproximation.MaximumValue(),
        AdaptiveCrossApproximation.MimicryPivoting(rowpos, colpos),
        conv(),
    )
    colbuffer2 = zeros(Float64, nrow, 30)
    rowbuffer2 = zeros(Float64, 30, 30)
    npiv2, rows2, cols2 = iaca_col_mimic(Kc, colbuffer2, rowbuffer2, 30)
    @test npiv2 > 0
    @test norm(Kc[:, cols2] * inv(Kc[rows2, cols2]) * Kc[rows2, :] - Kc) / norm(Kc) < 8e-2

    iaca_row_tree = AdaptiveCrossApproximation.iACA(
        AdaptiveCrossApproximation.TreeMimicryPivoting(colpos, rowpos, rowtree),
        AdaptiveCrossApproximation.MaximumValue(),
        conv(),
    )
    npiv3, rows3, cols3 = iaca_row_tree(
        Kc, colbuffer, rowbuffer, 30; rowidcs=Vector(1:nrow), colidcs=Vector(1:ncol)
    )
    @test npiv3 > 0
    @test norm(Kc[:, cols3] * inv(Kc[rows3, cols3]) * Kc[rows3, :] - Kc) / norm(Kc) < 1.2e-1

    iaca_col_tree = AdaptiveCrossApproximation.iACA(
        AdaptiveCrossApproximation.MaximumValue(),
        AdaptiveCrossApproximation.TreeMimicryPivoting(rowpos, colpos, coltree),
        conv(),
    )
    npiv4, rows4, cols4 = iaca_col_tree(
        Kc, colbuffer2, rowbuffer2, 30; rowidcs=Vector(1:nrow), colidcs=Vector(1:ncol)
    )
    @test npiv4 > 0
    @test norm(Kc[:, cols4] * inv(Kc[rows4, cols4]) * Kc[rows4, :] - Kc) / norm(Kc) < 1.2e-1
end

@testset "iACA contract: unsupported combinations" begin
    Random.seed!(8)
    rowpos = [@SVector rand(3) for _ in 1:20]
    colpos = [@SVector rand(3) for _ in 1:22] .+ Scalar(SVector(3.5, 0.0, 0.0))

    @test_throws MethodError AdaptiveCrossApproximation.iACA(
        AdaptiveCrossApproximation.MaximumValue(),
        AdaptiveCrossApproximation.MaximumValue(),
        AdaptiveCrossApproximation.FNormExtrapolator(
            AdaptiveCrossApproximation.iFNormEstimator(1e-4)
        ),
    )(
        Vector(1:20), Vector(1:22); maxrank=15
    )

    @test_throws MethodError AdaptiveCrossApproximation.iACA(
        AdaptiveCrossApproximation.MimicryPivoting(colpos, rowpos),
        AdaptiveCrossApproximation.MaximumValue(),
        AdaptiveCrossApproximation.iFNormEstimator(1e-4),
    )(
        Vector(1:20), Vector(1:22); maxrank=15
    )

    @test_throws MethodError AdaptiveCrossApproximation.iACA(
        AdaptiveCrossApproximation.MaximumValue(),
        AdaptiveCrossApproximation.MimicryPivoting(rowpos, colpos),
        AdaptiveCrossApproximation.FNormEstimator(1e-4),
    )(
        Vector(1:20), Vector(1:22); maxrank=15
    )
end
