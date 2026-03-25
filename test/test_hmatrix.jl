using AdaptiveCrossApproximation
using H2Trees
using LinearAlgebra
using Random
using StaticArrays
using Test
Random.seed!(1234)

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

struct myfct32 end
Base.eltype(::myfct32) = Float32
function (::myfct32)(x, y)
    if x == y
        return 0.0
    else
        return inv(norm(x - y))
    end
end
fct32 = myfct32()

@testset "H-Matrix" begin
    tpts = [SVector(rand(), rand(), 0.0) for i in 1:200]
    farpts = [SVector(rand() + 3.0, rand(), 0.0) for i in 1:210]
    farnearpts = [SVector(rand() + 2.0, rand(), 0.0) for i in 1:210]

    for mesh in [(tpts, tpts), (tpts, farpts), (tpts, farnearpts)]
        for spaceordering in [
            AdaptiveCrossApproximation.PermuteSpaceInPlace(),
            AdaptiveCrossApproximation.PreserveSpaceOrder(),
        ]
            for tol in [1e-2, 1e-4, 1e-6]
                tree = TwoNTree(
                    mesh[1], mesh[2], 1 / 2^10; testminvalues=100, trialminvalues=100
                )

                mat = AdaptiveCrossApproximation.HMatrix(
                    fct, mesh[1], mesh[2], tree; spaceordering=spaceordering, tol=tol
                )
                A = [fct(x, y) for x in mesh[1], y in mesh[2]]
                @test norm(Matrix(mat) - A) / norm(A) < tol
                x = rand(eltype(mat), size(mat, 2))
                @test norm(mat * x - A * x) / norm(A * x) < tol
                x = rand(eltype(mat), size(mat, 1))
                @test norm(transpose(mat) * x - transpose(A) * x) / norm(transpose(A) * x) <
                    tol
                @test norm(adjoint(mat) * x - adjoint(A) * x) / norm(adjoint(A) * x) < tol

                mat = AdaptiveCrossApproximation.HMatrix(
                    fct32, mesh[1], mesh[2], tree; spaceordering=spaceordering, tol=tol
                )
                @test eltype(mat) == Float32
                A = [fct32(x, y) for x in mesh[1], y in mesh[2]]
                @test norm(Matrix(mat) - A) / norm(A) < tol

                x = rand(eltype(mat), size(mat, 2))
                y = mat * x
                @test eltype(y) == Float32
                @test norm(y - A * x) / norm(A * x) < tol

                x = rand(eltype(mat), size(mat, 1))
                y = transpose(mat) * x
                @test eltype(y) == Float32
                @test norm(y - transpose(A) * x) / norm(transpose(A) * x) < tol
                y = adjoint(mat) * x
                @test eltype(y) == Float32
                @test norm(y - adjoint(A) * x) / norm(adjoint(A) * x) < tol
            end
        end
    end
end
