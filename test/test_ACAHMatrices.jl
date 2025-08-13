using BEAST
using HMatrices
using CompScienceMeshes
using AdaptiveCrossApproximation
using Random
using Test
using LinearAlgebra

Γ = meshsphere(1.0, 0.2)
op = Helmholtz3D.singlelayer()
spaceX = lagrangecxd0(Γ)
dim = length(spaceX.pos)
xclt = ClusterTree(spaceX.pos)

K = HMatrices.KernelMatrix(op, spaceX, spaceX);

rtols = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14]
tst_vec = rand(dim)

fullmat = BEAST.assemble(op, spaceX, spaceX)
trueResult = fullmat * tst_vec;

# PivStrat :: MaximumValue
for rtol in rtols
    aca = ACA(; convergence=FNormEstimator(0.0, rtol))
    hmat = HMatrices.assemble_hmatrix(K; comp=aca)
    @test size(hmat, 1) == size(fullmat, 1)
    @test size(hmat, 2) == size(fullmat, 2)
    @test norm(hmat * tst_vec - trueResult) / norm(trueResult) ≈ 0 atol = rtol
end
# PivStrat :: leja2
for rtol in rtols
    aca = ACA(; rowpivoting=Leja2(spaceX.pos), convergence=FNormEstimator(0.0, rtol))
    hmat = HMatrices.assemble_hmatrix(K; comp=aca)
    @test size(hmat, 1) == size(fullmat, 1)
    @test size(hmat, 2) == size(fullmat, 2)
    @test norm(hmat * tst_vec - trueResult) / norm(trueResult) ≈ 0 atol = rtol

    aca = ACA(; columnpivoting=Leja2(spaceX.pos), convergence=FNormEstimator(0.0, rtol))
    hmat = HMatrices.assemble_hmatrix(K; comp=aca)
    @test size(hmat, 1) == size(fullmat, 1)
    @test size(hmat, 2) == size(fullmat, 2)
    @test norm(hmat * tst_vec - trueResult) / norm(trueResult) ≈ 0 atol = rtol
end

# PivStrat :: FillDistance
for rtol in rtols
    aca = ACA(; rowpivoting=FillDistance(spaceX.pos), convergence=FNormEstimator(0.0, rtol))
    hmat = HMatrices.assemble_hmatrix(K; comp=aca)
    @test size(hmat, 1) == size(fullmat, 1)
    @test size(hmat, 2) == size(fullmat, 2)
    @test norm(hmat * tst_vec - trueResult) / norm(trueResult) ≈ 0 atol = rtol

    aca = ACA(;
        columnpivoting=FillDistance(spaceX.pos), convergence=FNormEstimator(0.0, rtol)
    )
    hmat = HMatrices.assemble_hmatrix(K; comp=aca)
    @test size(hmat, 1) == size(fullmat, 1)
    @test size(hmat, 2) == size(fullmat, 2)
    @test norm(hmat * tst_vec - trueResult) / norm(trueResult) ≈ 0 atol = rtol
end

# PivStrat :: CombinedPivStrat
for rtol in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]#, 1e-12, 1e-14]
    Random.seed!(1)
    cc1 = AdaptiveCrossApproximation.FNormEstimator(rtol)
    cc2 = AdaptiveCrossApproximation.RandomSampling(
        Float64; nsamples=100, factor=1.0, tol=rtol
    )
    convergence = AdaptiveCrossApproximation.CombinedConvCrit([cc1, cc2], zeros(Bool, 2))

    ps1 = MaximumValue()
    ps2 = AdaptiveCrossApproximation.RandomSamplingPivoting(cc2, 1)
    rp = AdaptiveCrossApproximation.CombinedPivStrat(convergence, [ps1, ps2])
    aca = ACA(; rowpivoting=rp, convergence=convergence)
    hmat = HMatrices.assemble_hmatrix(K; comp=aca)
    @test size(hmat, 1) == size(fullmat, 1)
    @test size(hmat, 2) == size(fullmat, 2)
    @test norm(hmat * tst_vec - trueResult) / norm(trueResult) ≈ 0 atol = rtol
end
