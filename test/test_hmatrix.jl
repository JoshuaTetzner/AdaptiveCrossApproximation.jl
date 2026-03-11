using H2Trees
using StaticArrays
using AdaptiveCrossApproximation
using LinearAlgebra
using Test

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
spts = [(@SVector rand(3)) + SVector(3.0, 0.0, 0.0) for i in 1:400]

for mesh in [(pts, pts), (tpts, tpts), (tpts, spts)]
    for tol in [1e-2, 1e-4, 1e-6]
        for space_ordering in [
            AdaptiveCrossApproximation.PreserveSpaceOrder(),
            AdaptiveCrossApproximation.PermuteSpaceInPlace(),
        ]
            tree = TwoNTree(mesh[1], mesh[2], 1 / 2^10; minvaluestest=100, minvaluestrial=100)
            @time mat = AdaptiveCrossApproximation.HMatrix(fct, mesh[1], mesh[2], tree; space_ordering=space_ordering, tol=tol)
            local A = [fct(x, y) for x in mesh[1], y in mesh[2]]
            @test norm(Matrix(mat) - A) / norm(A) < tol
        end
    end
end


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
@time mat = AdaptiveCrossApproximation.HMatrix(fct, tpts, spts, tree; tol=1e-2)

@test eltype(mat) == Float32
y = mat * rand(Float32, 400)
@test eltype(y) == Float32
