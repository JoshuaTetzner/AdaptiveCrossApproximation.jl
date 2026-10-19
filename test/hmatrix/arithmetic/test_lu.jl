using AdaptiveCrossApproximation
using H2Trees
using LinearAlgebra
using Random
using Test

function lu_test_dense(W)
    A = zeros(eltype(W), size(W))
    csr = W.csr
    for k in eachindex(csr.storageidx)
        rows, cols = AdaptiveCrossApproximation._productranges(W, k)
        A[rows, cols] =
            csr.isfar[k] ? Matrix(W.far[csr.storageidx[k]]) : W.near[csr.storageidx[k]]
    end
    return A
end

function lu_test_factors(F)
    W, csr = F.view, F.view.csr
    L, U = zeros(eltype(W), size(W)), zeros(eltype(W), size(W))
    for k in eachindex(csr.storageidx)
        row, col = W.entryrows[k], csr.colnodes[k]
        rows, cols = W.rowranges[row], W.colranges[col]
        block = csr.isfar[k] ? Matrix(W.far[csr.storageidx[k]]) : W.near[csr.storageidx[k]]
        if row == col
            L[rows, cols] = UnitLowerTriangular(block)
            U[rows, cols] = UpperTriangular(block)
        elseif last(cols) < first(rows)
            L[rows, cols] = block
        else
            U[rows, cols] = block
        end
    end
    return L, U
end

@testset "lu.jl" begin
    ACA = AdaptiveCrossApproximation
    rng = MersenneTwister(48117)
    local core
    for T in (Float64, ComplexF64),
        ordering in (ACA.PreserveSpaceOrder(), ACA.PermuteSpaceInPlace())

        core, _ = triangular_test_fixture(rng, T; n=23, rank=2, ordering, factorize=false)
        A = Matrix(core)
        F = lu(core)
        tolerance = 1e-8
        @testset "lu ($T, $(typeof(ordering)))" begin
            @test F isa HMatrixLU
            @test Matrix(core) == A
            @test issuccess(F)
            @test F.converged
            @test !F.rank_limited
            @test any(
                Int(pivot) != i for leaf in F.leaves for (i, pivot) in enumerate(leaf.ipiv)
            )

            PA = lu_test_dense(ACA._TreeView(core))
            ACA._pivot!(PA, F.lower)
            L, U = lu_test_factors(F)
            @test norm(PA - L * U) <= tolerance * norm(A)
        end

        b = randn(rng, T, size(A, 1))
        B = randn(rng, T, size(A, 1), 3)
        x = F \ b
        @testset "ldiv! / SolveWorkspace ($T, $(typeof(ordering)))" begin
            @test norm(A * x - b) <= tolerance * norm(b)
            S = ACA.SolveWorkspace(F)
            @test norm(A * (S \ B) - B) <= tolerance * norm(B)
            @test norm(transpose(A) * (transpose(S) \ b) - b) <= tolerance * norm(b)
            @test norm(A' * (S' \ b) - b) <= tolerance * norm(b)
            y = similar(b)
            @test ldiv!(y, S, b) === y
            @test y ≈ x
            z = copy(b)
            @test ldiv!(S, z) === z
            @test z ≈ x
            @test ACA.SolveWorkspace(S).scratch !== S.scratch
            if T <: Real
                c = complex.(b, reverse(b))
                @test norm(A * (F \ c) - c) <= tolerance * norm(c)
            end
        end

        @testset "lu! / LUWorkspace ($T, $(typeof(ordering)))" begin
            # lu! factorizes in place; a workspace serves every matrix with the same
            # partition, and earlier factorizations stay valid.
            G = copy(core)
            workspace = ACA.LUWorkspace(G)
            H = lu!(G, workspace)
            @test Matrix(G) != A
            @test H \ b ≈ x
            scaled = lu(2 * core, workspace)
            @test scaled \ b ≈ x / 2
            @test H \ b ≈ x
            @test F \ b ≈ x
            other, _ = triangular_test_fixture(
                rng, T; n=23, rank=2, ordering, factorize=false
            )
            @test_throws ArgumentError lu(other, workspace)
        end

        @testset "_rungraph! (serial/parallel arithmetic, $T)" begin
            # Parallel execution reproduces the serial results exactly.
            blocks(M) = (M.near, [(R.U, R.V) for R in M.far])
            serial = lu(core, ACA.LUWorkspace(core; ntasks=1))
            parallel = lu(core, ACA.LUWorkspace(core; ntasks=3))
            @test blocks(parallel.view) == blocks(serial.view)
            @test [leaf.ipiv for leaf in parallel.leaves] == [leaf.ipiv for leaf in serial.leaves]
            product(ntasks) = mul!(
                copy(core),
                core,
                core,
                2,
                1,
                ACA.ProductWorkspace(core, core, core; ntasks),
            )
            @test blocks(product(3)) == blocks(product(1))
            scaled = 2 * core
            sum(ntasks) =
                axpy!(0.5, scaled, copy(core), ACA.AdditionWorkspace(core, scaled; ntasks))
            @test blocks(sum(3)) == blocks(sum(1))
        end
    end
    @testset "LUWorkspace (validation)" begin
        @test_throws ArgumentError ACA.LUWorkspace(core; ntasks=0)
    end

    @testset "lu (singular diagonal blocks)" begin
        core, _ = triangular_test_fixture(rng, Float64; n=17, factorize=false)
        W = ACA._TreeView(core)
        k = findfirst(k -> W.entryrows[k] == W.csr.colnodes[k], eachindex(W.csr.storageidx))
        fill!(core.near[core.csr.storageidx[k]], 0)
        @test_throws LeafLUFailure lu(core)
        F = lu(core; check=false)
        @test !issuccess(F)
        @test_throws ArgumentError F \ ones(17)
        @test_throws LeafLUFailure lu(core, ACA.LUWorkspace(core; ntasks=3))
        @test !issuccess(lu(core, ACA.LUWorkspace(core; ntasks=3); check=false))
    end
end
