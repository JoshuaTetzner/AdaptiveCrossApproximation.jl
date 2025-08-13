using BEAST
using HMatrices
using CompScienceMeshes
using AdaptiveCrossApproximation
using Test
using LinearAlgebra

Γ = meshsphere(1.0, 0.2)
op = Helmholtz3D.singlelayer()
spaceX = lagrangecxd0(Γ)
dim = length(spaceX.pos)
xclt = ClusterTree(spaceX.pos)

K = HMatrices.KernelMatrix(op, spaceX, spaceX);

rtols = [10.0^i for i in collect(-4:-1:-10)]
tst_vec = rand(dim)

fullmat = BEAST.assemble(op, spaceX, spaceX)
trueResult = fullmat * tst_vec;
##

for rtol in rtols
    K = HMatrices.KernelMatrix(op, spaceX, spaceX)
    aca = ACA(; convergence=FNormEstimator(0.0, rtol))
    hmat = HMatrices.assemble_hmatrix(K; comp=aca)
    @test size(hmat, 1) == size(fullmat, 1)
    @test size(hmat, 2) == size(fullmat, 2)
    @test norm(hmat * tst_vec - trueResult) / norm(trueResult) ≈ 0 atol = rtol
end
