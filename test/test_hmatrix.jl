using H2Trees
using StaticArrays
using AdaptiveCrossApproximation
using LinearAlgebra
using Test
##
struct myfct end
Base.eltype(::myfct) = Float64
function (::myfct)(x, y)
    if x == y
        return 0.0
    else
        return inv(norm(x - y))
    end
end
fct = myfct()

pts = [@SVector rand(3) for i in 1:2000]
tpts = [@SVector rand(3) for i in 1:201]
spts = [(@SVector rand(3)) + SVector(3.0, 0.0, 0.0) for i in 1:300]
##
for mesh in [(pts, pts), (tpts, tpts), (tpts, spts)]
    for spaceordering in [
        AdaptiveCrossApproximation.PermuteSpaceInPlace(),
        AdaptiveCrossApproximation.PreserveSpaceOrder(),
    ]
        for tol in [1e-2, 1e-4, 1e-6]
            tree = TwoNTree(
                mesh[1], mesh[2], 1 / 2^10; minvaluestest=100, minvaluestrial=100
            )
            mat = AdaptiveCrossApproximation.HMatrix(
                fct, mesh[1], mesh[2], tree; spaceordering=spaceordering, tol=tol
            )
            local A = [fct(x, y) for x in mesh[1], y in mesh[2]]
            @test norm(Matrix(mat) - A) / norm(A) < tol
            x = rand(eltype(mat), size(mat, 2))
            @test norm(mat * x - A * x) / norm(A * x) < tol
            x = rand(eltype(mat), size(mat, 1))
            @test norm(transpose(mat) * x - transpose(A) * x) / norm(transpose(A) * x) < tol
            @test norm(adjoint(mat) * x - adjoint(A) * x) / norm(adjoint(A) * x) < tol
        end
    end
end
##
struct myfct32 end
Base.eltype(::myfct32) = Float32
function (::myfct32)(x, y)
    if x == y
        return 0.0
    else
        return inv(norm(x - y))
    end
end
fct = myfct32()

tree = TwoNTree(tpts, spts, 1 / 2^10; minvaluestest=100, minvaluestrial=100)
@time mat = AdaptiveCrossApproximation.HMatrix(
    fct,
    tpts,
    spts,
    tree;
    spaceordering=AdaptiveCrossApproximation.PermuteSpaceInPlace(),
    tol=1e-2,
)
@test eltype(mat) == Float32
y = mat * rand(Float32, 300)
@test eltype(y) == Float32

@time mat = AdaptiveCrossApproximation.HMatrix(
    fct,
    tpts,
    spts,
    tree;
    spaceordering=AdaptiveCrossApproximation.PreserveSpaceOrder(),
    tol=1e-2,
)
@test eltype(mat) == Float32
y = mat * rand(Float32, 300)
@test eltype(y) == Float32
