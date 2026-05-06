using AdaptiveCrossApproximation
using LinearAlgebra
using Random
using StaticArrays
using Test

@testset "ACA" begin
    Random.seed!(1234)

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

@testset "ACA Special Cases" begin
    Random.seed!(1234)
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
end

##
using AdaptiveCrossApproximation
using BEAST
using CompScienceMeshes
using StaticArrays
using FastBEAST
using LinearAlgebra

λ = 1.0
gamma = im * 2 * pi / λ
alpha = -gamma
beta = -1 / gamma
m1 = meshrectangle(1.0, 1.0, 0.1)
space1 = raviartthomas(m1)
length(space1)
m2 = CompScienceMeshes.translate(meshrectangle(1.0, 1.0, 0.1), SVector(0.0, 0.0, -2.0))
space2 = raviartthomas(m2)
length(space2)
##
op = Maxwell3D.singlelayer(; wavenumber=2 * pi / λ)
ϕ = Maxwell3D.singlelayer(; gamma=gamma, alpha=im * 0.0, beta=beta)
A = Maxwell3D.singlelayer(; gamma=gamma, alpha=alpha, beta=0.0 * im)

##
T = assemble(op, space1, space2)
T_ϕ = assemble(ϕ, space1, space2)
T_A = assemble(A, space1, space2)
##
U, V = AdaptiveCrossApproximation.aca(T; tol=1e-3, maxrank=40)
norm(U * V - T) / norm(T)

U_ϕ, V_ϕ = AdaptiveCrossApproximation.aca(T_ϕ; tol=1e-3, maxrank=40)
norm(U_ϕ * V_ϕ - T_ϕ) / norm(T_ϕ)

U_A, V_A = AdaptiveCrossApproximation.aca(T_A; tol=1e-3, maxrank=40)
norm(U_A * V_A - T_A) / norm(T_A)

##
using ParallelKMeans
using H2Trees

tree = KMeansTree(
    space2.pos, 2; minvalues=50, updateradii=H2Trees.unsafemaxradiusboundingsphere
)
nodes = collect(H2Trees.LevelIterator(tree, 2))
edges = getedges(m2)

iaca = iACA(
    MaximumValue(),
    AdaptiveCrossApproximation.TreeMimicryPivoting2(space1.pos, space2.pos, edges, tree),
    FNormExtrapolator(iFNormEstimator(1e-3)),
)

iaca = iaca([1], [1], 40)

colbuffer = zeros(ComplexF64, length(space1), 40)
rowbuffer = zeros(ComplexF64, 40, 40)
rows = zeros(Int, 40)
cols = zeros(Int, 40);

##
npivots, r, c = iaca(
    T, colbuffer, rowbuffer, rows, cols, collect(1:length(space1)), nodes, 40
)
norm(T - T[:, c] * inv(T[r, c]) * T[r, :]) / norm(T)

npivots, r, c = iaca(
    T_ϕ, colbuffer, rowbuffer, rows, cols, collect(1:length(space1)), nodes, 40
)
norm(T_ϕ - T_ϕ[:, c] * inv(T_ϕ[r, c]) * T_ϕ[r, :]) / norm(T_ϕ)

npivots, r, c = iaca(
    T_A, colbuffer, rowbuffer, rows, cols, collect(1:length(space1)), nodes, 40
)
norm(T_A - T_A[:, c] * inv(T_A[r, c]) * T_A[r, :]) / norm(T_A)
