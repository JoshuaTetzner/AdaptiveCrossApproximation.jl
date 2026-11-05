using AdaptiveCrossApproximation
using AdaptiveCrossApproximation: LowRankMatrix
using BEAST
using CompScienceMeshes
using H2Trees
using HFactory: HFactory
using LinearAlgebra
using Random
using StaticArrays
using Test

struct HelmholtzKernel end
Base.eltype(::HelmholtzKernel) = ComplexF64
(::HelmholtzKernel)(x, y) = x == y ? zero(ComplexF64) : cis(4 * norm(x - y)) / norm(x - y)

function clusterindex(points; minvalues)
    builder = H2Trees.TwoNTreeBuilder(; minhalfsize=1 / 2^10, minvalues)
    return HFactory.ClusterIndex(H2Trees.buildtree(points; builder), length(points))
end

alwaysnear(testtree, trialtree, a, b) = true
alwaysfar(testtree, trialtree, a, b) = false

# Rectangular point-kernel problem with H2Trees cluster permutations.
function hfactory_fixture(op; m=1200, n=900, isnear=HFactory.isnear())
    rng = MersenneTwister(31)
    testpoints = [SVector(rand(rng), rand(rng), 0.0) for _ in 1:m]
    trialpoints = [SVector(rand(rng) + 0.25, rand(rng), 0.0) for _ in 1:n]
    index = HFactory.BlockIndex(
        clusterindex(testpoints; minvalues=32),
        clusterindex(trialpoints; minvalues=32);
        isnear,
    )
    kernel = AbstractKernelMatrix(op, testpoints, trialpoints)
    dense = [op(x, y) for x in testpoints, y in trialpoints]
    return kernel, dense, index, testpoints, trialpoints
end

# Calls each block of `H` with its row and column indices, its kind and its payload.
function foreach_block(f, H)
    index = HFactory.blockindex(H)
    for row in eachindex(index.test.ranges)
        rownode = row + H2Trees.root(index.test.tree) - 1
        for k in HFactory.rowblocks(index, rownode)
            rows, cols = HFactory.blockindices(index, rownode, index.csr.colnodes[k])
            p = index.csr.storageidx[k]
            far = index.csr.isfar[k]
            f(rows, cols, far, far ? H.compressed[p] : H.dense[p])
        end
    end
end

# Counts the block-local preparations of the wrapped kernel matrix.
struct PreparationCounter{T,K} <: AbstractKernelMatrix{T}
    kernel::K
    count::Threads.Atomic{Int}
end
PreparationCounter(kernel) =
    PreparationCounter{eltype(kernel),typeof(kernel)}(kernel, Threads.Atomic{Int}(0))
Base.size(K::PreparationCounter, dim...) = size(K.kernel, dim...)
(K::PreparationCounter)(block, rows, cols) = K.kernel(block, rows, cols)
function AdaptiveCrossApproximation.localkernelmatrix(K::PreparationCounter, rows, cols)
    Threads.atomic_add!(K.count, 1)
    return AdaptiveCrossApproximation.localkernelmatrix(K.kernel, rows, cols)
end

function sampling_allocations(buf, A, i, j)
    AdaptiveCrossApproximation.nextrc!(buf, A, i, j)
    return @allocated AdaptiveCrossApproximation.nextrc!(buf, A, i, j)
end

relerr(a, b) = norm(a - b) / norm(b)

function matvec_allocations(y, A, x)
    mul!(y, A, x)
    return @allocated mul!(y, A, x)
end

@testset "hfactory.jl" begin
    tol = 1e-6
    far = LowRankCompression(ACA(; tol); maxrank=60)
    scheme = HFactory.HMatrixConstruction(nothing, far)

    @testset "LowRankCompression" begin
        @test LowRankCompression() isa LowRankCompression{<:ACA}
        @test LowRankCompression().maxrank == 40
    end

    @testset "Construction ($T)" for (op, T) in
                                     ((fct, Float64), (HelmholtzKernel(), ComplexF64))
        kernel, dense, index = hfactory_fixture(op)
        m, n = size(dense)
        @test index.test.perm != 1:m && index.trial.perm != 1:n
        @test HFactory.compressedblocktype(far, kernel) === LowRankMatrix{T}

        H = @inferred HFactory.assemble(kernel, index, scheme)
        @test H isa HFactory.HMatrixCore{T,LowRankMatrix{T}}
        @test size(H) == (m, n)
        @test length(H.compressed) > 100 && length(H.dense) > 100
        @test 1 < maximum(rank, H.compressed) < far.maxrank
        foreach_block(H) do rows, cols, isfar, B
            @test size(B) == (length(rows), length(cols))
            if isfar
                @test relerr(Matrix(B), dense[rows, cols]) < 10tol
            else
                @test B == dense[rows, cols]
            end
        end

        # Plain matrices use the same compressor and HFactory's near-field default.
        Hdense = HFactory.assemble(dense, index, scheme)
        @test Hdense.dense == H.dense
        @test all(zip(Hdense.compressed, H.compressed)) do (A, B)
            return Matrix(A) ≈ Matrix(B)
        end

        @testset "MVP with working type $S" for S in unique((T, ComplexF64))
            ws = HFactory.MatVecWorkspace(H, S)
            rng = MersenneTwister(5)
            x, xt = rand(rng, S, n), rand(rng, S, m)
            @test relerr(ws * x, dense * x) < 10tol
            @test relerr(transpose(ws) * xt, transpose(dense) * xt) < 10tol
            @test relerr(ws' * xt, dense' * xt) < 10tol
            y0 = rand(rng, S, m)
            α, β = S(2), S(-1)
            @test relerr(mul!(copy(y0), ws, x, α, β), α * dense * x + β * y0) < 10tol

            y, yt = zeros(S, m), zeros(S, n)
            @inferred mul!(y, ws, x)
            @test matvec_allocations(y, ws, x) == 0
            @test matvec_allocations(yt, transpose(ws), xt) == 0
            @test matvec_allocations(yt, ws', xt) == 0
        end
    end

    @testset "Scratch reuse and factor ownership" begin
        kernel, dense, index = hfactory_fixture(HelmholtzKernel())
        workspace = HFactory.ConstructionWorkspace(kernel, index, scheme)
        (; U, V) = workspace.far
        @test size(U, 1) < size(dense, 1) && size(V, 2) < size(dense, 2)
        H = HFactory.assemble(kernel, index, scheme, workspace)
        @test all(B -> !Base.mightalias(B.U, U) && !Base.mightalias(B.V, V), H.compressed)
        factors = [(copy(B.U), copy(B.V)) for B in H.compressed]

        # Reassembly overwrites the buffers but neither the earlier factors nor itself.
        H2 = HFactory.assemble(kernel, index, scheme, workspace)
        @test all(((B, (U0, V0)),) -> B.U == U0 && B.V == V0, zip(H.compressed, factors))
        @test all(((A, B),) -> A.U == B.U && A.V == B.V, zip(H.compressed, H2.compressed))
        for rownode in H2Trees.treeindex(index.test.tree).depthfirstnodes
            for k in HFactory.rowblocks(index, rownode)
                index.csr.isfar[k] || continue
                B = HFactory.block(H, k)
                B === first(H.compressed) || continue
                C = @inferred HFactory.assemble_far!(
                    kernel, index, rownode, index.csr.colnodes[k], far, workspace.far
                )
                @test C.U == B.U && C.V == B.V
            end
        end
        fill!(U, NaN)
        fill!(V, NaN)
        x = rand(MersenneTwister(2), ComplexF64, size(dense, 2))
        @test relerr(HFactory.MatVecWorkspace(H) * x, dense * x) < 10tol

        # Multiplication neither changes the factors nor shares scratch between workspaces.
        ws1, ws2 = HFactory.MatVecWorkspace(H), HFactory.MatVecWorkspace(H)
        @test length(ws1.scratch) == maximum(rank, H.compressed)
        @test ws1.scratch !== ws2.scratch
        b = dense * x
        mul!(b, ws1, x)
        mul!(x, transpose(ws1), b)
        mul!(x, ws1', b)
        @test all(((B, (U0, V0)),) -> B.U == U0 && B.V == V0, zip(H.compressed, factors))
    end

    @testset "Block kernels" begin
        rng = MersenneTwister(9)
        z = zeros(ComplexF64, 4)
        for B in (
            LowRankMatrix(rand(rng, 7, 3), rand(rng, 3, 5)),
            LowRankMatrix(zeros(7, 0), zeros(0, 5)),
        )
            M = Matrix(B)
            for (op, A) in ((Val(:N), M), (Val(:T), transpose(M)), (Val(:C), M'))
                x = rand(rng, ComplexF64, size(A, 2))
                y = fill(complex(NaN), size(A, 1))
                @test HFactory.blockmul!(y, B, view(x, :), 2, false, z, op) ≈ 2 * A * x
                y0 = rand(rng, ComplexF64, size(A, 1))
                @test HFactory.blockmul!(copy(y0), B, x, 2, 3, z, op) ≈ 2 * A * x + 3 * y0
            end
        end
    end

    @testset "Empty partitions" begin
        kernel, dense, index = hfactory_fixture(fct; m=300, n=200, isnear=alwaysnear)
        H = HFactory.assemble(kernel, index, scheme)
        @test isempty(H.compressed) && H.compressed isa Vector{LowRankMatrix{Float64}}
        x = rand(MersenneTwister(3), size(dense, 2))
        @test HFactory.MatVecWorkspace(H) * x ≈ dense * x

        # The overlapping root block is not low rank; it is truncated, not stored dense.
        kernel, dense, index = hfactory_fixture(fct; m=300, n=200, isnear=alwaysfar)
        H = @test_logs (:warn, "Maximum rank block") HFactory.assemble(
            kernel, index, scheme
        )
        @test isempty(H.dense) && length(H.compressed) == 1
        @test size(only(H.compressed)) == (300, 200)
        @test rank(only(H.compressed)) == far.maxrank
    end

    @testset "BEAST kernel matrix" begin
        # DoubleNumQStrat compresses through LocalBEASTKernelMatrix.
        quadstrat = BEAST.DoubleNumQStrat(2, 3)
        plate = meshrectangle(1.0, 1.0, 0.05)
        Y = lagrangec0d1(plate)
        X = lagrangec0d1(translate(meshrectangle(1.0, 0.8, 0.06), [0.0, 0.0, 0.3]))
        op = Helmholtz3D.singlelayer(; wavenumber=2.0)
        kernel = AbstractKernelMatrix(op, Y, X; matrixdata=quadstrat)
        @test AdaptiveCrossApproximation.localkernelmatrix(kernel, [1], [1]) isa
            AbstractKernelMatrix
        index = HFactory.BlockIndex(
            clusterindex(BEAST.positions(Y); minvalues=16),
            clusterindex(BEAST.positions(X); minvalues=16),
        )
        H = HFactory.assemble(kernel, index, scheme)
        @test !isempty(H.compressed) && !isempty(H.dense)
        dense = assemble(op, Y, X; quadstrat)
        x = rand(MersenneTwister(6), ComplexF64, size(dense, 2))
        @test relerr(HFactory.MatVecWorkspace(H) * x, dense * x) < 10tol

        # Each far block prepares its local kernel exactly once.
        counted = PreparationCounter(kernel)
        Hcounted = HFactory.assemble(counted, index, scheme)
        @test counted.count[] == length(Hcounted.compressed)
        @test all(
            ((A, B),) -> A.U == B.U && A.V == B.V, zip(H.compressed, Hcounted.compressed)
        )

        # One prepared local kernel serves repeated ACA runs without allocating samples.
        extension = Base.get_extension(HFactory, :HFactoryBEAST)
        foreach_block(H) do rows, cols, isfar, B
            B === first(H.compressed) || return nothing
            localmatrix = AdaptiveCrossApproximation.localkernelmatrix(kernel, rows, cols)
            @test localmatrix isa extension.LocalBEASTKernelMatrix
            compressor = ACA(; tol)
            factors = map(1:2) do _
                U = zeros(ComplexF64, length(rows), far.maxrank)
                V = zeros(ComplexF64, far.maxrank, length(cols))
                r = compressor(localmatrix, U, V, far.maxrank; rowidcs=rows, colidcs=cols)
                return U[:, 1:r], V[1:r, :]
            end
            @test factors[1] == factors[2] == (B.U, B.V)
            @test relerr(prod(factors[1]), dense[rows, cols]) < 10tol
            rowbuf = zeros(ComplexF64, 1, length(cols))
            @test sampling_allocations(rowbuf, localmatrix, rows[1], cols) == 0
            colbuf = zeros(ComplexF64, length(rows), 1)
            @test sampling_allocations(colbuf, localmatrix, rows, cols[1]) == 0
        end
    end

    @testset "Comparison with HMatrix" begin
        kernel, dense, index, testpoints, trialpoints = hfactory_fixture(fct)
        H = HFactory.assemble(kernel, index, scheme)
        builder = H2Trees.TwoNTreeBuilder(; minhalfsize=1 / 2^10, minvalues=32)
        tree = H2Trees.buildtree(
            testpoints,
            trialpoints;
            builder=H2Trees.BlockTreeBuilder(; test=builder, trial=builder),
        )
        reference = AdaptiveCrossApproximation.HMatrix(
            fct,
            testpoints,
            trialpoints,
            tree;
            tol,
            compressor=far.compressor,
            maxrank=far.maxrank,
            verbose=false,
        )
        x = rand(MersenneTwister(4), size(dense, 2))
        @test relerr(HFactory.MatVecWorkspace(H) * x, reference * x) < 10tol
    end
end
