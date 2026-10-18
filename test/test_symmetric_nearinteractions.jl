using BEAST
using CompScienceMeshes
using ParallelKMeans

function symmetric_test_tree(points; minvalues)
    return H2Trees.buildtree(
        points; builder=H2Trees.TwoNTreeBuilder(; minhalfsize=0.1, minvalues)
    )
end

mutable struct CountedSymmetricKernel{T}
    evaluations::Int
end
Base.eltype(::CountedSymmetricKernel{T}) where {T} = T
function (kernel::CountedSymmetricKernel{T})(x, y) where {T}
    kernel.evaluations += 1
    return T(1 / (1 + norm(x - y))) * (1 + 0.5im)
end

struct TypedSymmetricKernel{T}
    scale::T
end
Base.eltype(::TypedSymmetricKernel{T}) where {T} = T
(kernel::TypedSymmetricKernel{T})(x, y) where {T} = T(kernel.scale / (1 + norm(x - y)))

function csrrows(csr)
    rows = Vector{Int}(undef, length(csr))
    for row in 1:(length(csr.rowptr) - 1)
        rows[csr.rowptr[row]:(csr.rowptr[row + 1] - 1)] .= row
    end
    return rows
end

@testset "Symmetric H-matrix storage" begin
    points = [SVector(Float64(i), 0.0) for i in 1:12]
    dense = [(1 + 0.5im) / (1 + norm(x - y)) for x in points, y in points]
    cluster = symmetric_test_tree(points; minvalues=3)
    tree = H2Trees.BlockTree(cluster, cluster)
    kernel = CountedSymmetricKernel{ComplexF64}(0)
    hmat = AdaptiveCrossApproximation.HMatrix(
        kernel, points, tree; isnear=(treea, treeb, a, b) -> true, ntasks=1, verbose=false
    )

    @test kernel.evaluations == sum(length, hmat.near)
    @test kernel.evaluations < length(points)^2
    @test isempty(hmat.far)
    @test count(!, hmat.csr.isfar) == length(hmat.csr)
    @test sort(hmat.csc.csrmap) == collect(1:length(hmat.csr))

    rows = csrrows(hmat.csr)
    for csridx in findall(hmat.csr.transposed)
        partner = findfirst(
            i ->
                !hmat.csr.transposed[i] &&
                rows[i] == hmat.csr.colnodes[csridx] &&
                hmat.csr.colnodes[i] == rows[csridx] &&
                hmat.csr.storageidx[i] == hmat.csr.storageidx[csridx],
            eachindex(rows),
        )
        @test !isnothing(partner)
    end

    x = randn(ComplexF64, length(points))
    @test Matrix(hmat) ≈ dense
    @test hmat * x ≈ dense * x
    @test transpose(hmat) * x ≈ transpose(dense) * x
    @test adjoint(hmat) * x ≈ adjoint(dense) * x
end

@testset "Symmetric permuted storage" begin
    points = reverse([SVector(Float64(i), 0.0) for i in 1:24])
    cluster = symmetric_test_tree(points; minvalues=4)
    tree = H2Trees.BlockTree(cluster, cluster)
    hmat = AdaptiveCrossApproximation.HMatrix(
        CountedSymmetricKernel{ComplexF64}(0),
        points,
        tree;
        isnear=AdaptiveCrossApproximation.isnear(0.5),
        spaceordering=AdaptiveCrossApproximation.PermuteSpaceInPlace(),
        tol=1e-5,
        ntasks=1,
        verbose=false,
    )
    reference = [(1 + 0.5im) / (1 + norm(x - y)) for x in points, y in points]

    @test Matrix(hmat) ≈ reference rtol = 1e-4
    @test hmat * ones(ComplexF64, length(points)) ≈
        reference * ones(ComplexF64, length(points)) rtol = 1e-4
end

@testset "Symmetric mixed near/far storage" begin
    points = [SVector(Float64(i), 0.0) for i in 1:64]
    cluster = symmetric_test_tree(points; minvalues=4)
    tree = H2Trees.BlockTree(cluster, cluster)
    predicate = AdaptiveCrossApproximation.isnear(0.5)
    kernel = CountedSymmetricKernel{ComplexF64}(0)
    hmat = AdaptiveCrossApproximation.HMatrix(
        kernel, points, tree; isnear=predicate, tol=1e-5, ntasks=1, verbose=false
    )
    full = AdaptiveCrossApproximation.HMatrix(
        CountedSymmetricKernel{ComplexF64}(0),
        points,
        points,
        tree;
        isnear=predicate,
        tol=1e-5,
        ntasks=1,
        verbose=false,
    )
    reference = [(1 + 0.5im) / (1 + norm(x - y)) for x in points, y in points]

    @test !isempty(hmat.near)
    @test !isempty(hmat.far)
    @test count(identity, hmat.csr.transposed) > 0
    @test length(full.far) == 2length(hmat.far)
    @test Matrix(AdaptiveCrossApproximation.nearmatrix(hmat)) +
          Matrix(AdaptiveCrossApproximation.farmatrix(hmat)) ≈ Matrix(hmat)
    @test norm(Matrix(hmat) - Matrix(full)) / norm(Matrix(full)) < 1e-4
    @test norm(Matrix(hmat) - reference) / norm(reference) < 1e-4

    x = randn(ComplexF64, length(points))
    @test norm(hmat * x - reference * x) / norm(reference * x) < 1e-4
    @test norm(transpose(hmat) * x - transpose(reference) * x) /
          norm(transpose(reference) * x) < 1e-4
    @test norm(adjoint(hmat) * x - adjoint(reference) * x) / norm(adjoint(reference) * x) <
        1e-4

    parallel = AdaptiveCrossApproximation.HMatrix(
        CountedSymmetricKernel{ComplexF64}(0),
        points,
        tree;
        isnear=predicate,
        tol=1e-5,
        ntasks=2,
        verbose=false,
    )
    @test Matrix(parallel) ≈ Matrix(hmat)

    farself = AdaptiveCrossApproximation.HMatrix(
        CountedSymmetricKernel{ComplexF64}(0),
        points,
        tree;
        isnear=(treea, treeb, a, b) -> false,
        tol=1e-9,
        maxrank=length(points) + 1,
        ntasks=1,
        verbose=false,
    )
    @test isempty(farself.near)
    @test length(farself.far) == 1
    @test length(farself.csr) == 1
    @test norm(Matrix(farself) - reference) / norm(reference) < 1e-4
end

@testset "Symmetric BEAST assembly" begin
    mesh = CompScienceMeshes.meshsphere(1.0, 1.2)
    space = BEAST.raviartthomas(mesh)
    op = BEAST.Maxwell3D.singlelayer(; wavenumber=1.0)
    quadstrat = BEAST.DoubleNumSauterQstrat(8, 8, 0, 8, 8, 8)
    cluster = symmetric_test_tree(BEAST.positions(space); minvalues=8)
    tree = H2Trees.BlockTree(cluster, cluster)
    hmat = AdaptiveCrossApproximation.HMatrix(
        op,
        space,
        tree;
        isnear=AdaptiveCrossApproximation.isnear(2.0),
        nearmatrixdata=quadstrat,
        farmatrixdata=quadstrat,
        tol=1e-4,
        ntasks=1,
        verbose=false,
    )
    assembled = BEAST.assemble(op, space, space; quadstrat=quadstrat)
    @test norm(Matrix(hmat) - assembled) / norm(assembled) < 1e-4
    x = ones(ComplexF64, length(space))
    @test norm(hmat * x - assembled * x) / norm(assembled * x) < 1e-4
end

@testset "Symmetric scalar types" begin
    points = [SVector(Float64(i), 0.0) for i in 1:32]
    cluster = symmetric_test_tree(points; minvalues=4)
    tree = H2Trees.BlockTree(cluster, cluster)
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        scale = T <: Real ? T(1) : T(1 + 0.5im)
        kernel = TypedSymmetricKernel(scale)
        reference = [kernel(x, y) for x in points, y in points]
        hmat = AdaptiveCrossApproximation.HMatrix(
            kernel,
            points,
            tree;
            isnear=AdaptiveCrossApproximation.isnear(1.0),
            tol=1e-5,
            ntasks=1,
            verbose=false,
        )
        x = ones(T, length(points))
        @test eltype(hmat) == T
        @test norm(Matrix(hmat) - reference) / norm(reference) < 1e-4
        @test norm(hmat * x - reference * x) / norm(reference * x) < 1e-4
        @test transpose(hmat) * x ≈ hmat * x
        @test adjoint(hmat) * x ≈ conj.(hmat * x)
    end
end
