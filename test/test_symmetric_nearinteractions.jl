using BlockSparseMatrices
using OhMyThreads
using BEAST
using CompScienceMeshes
using ParallelKMeans

function symmetric_test_tree(points; minvalues)
    if isdefined(H2Trees, :TwoNTreeBuilder)
        return H2Trees.buildtree(
            points; builder=H2Trees.TwoNTreeBuilder(; minhalfsize=0.1, minvalues)
        )
    end
    return H2Trees.TwoNTree(points, 0.1; minvalues)
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
function (kernel::TypedSymmetricKernel{T})(x, y) where {T}
    return T(kernel.scale / (1 + norm(x - y)))
end

@testset "Symmetric H-matrix interactions" begin
    points = [SVector(Float64(i), 0.0) for i in 1:12]
    dense = [(1 + 0.5im) / (1 + norm(x - y)) for x in points, y in points]
    allnear = (treea, treeb, a, b) -> true

    let ordering = AdaptiveCrossApproximation.PreserveSpaceOrder()
        space = copy(points)
        cluster = symmetric_test_tree(space; minvalues=3)
        tree = H2Trees.BlockTree(cluster, cluster)
        kernel = CountedSymmetricKernel{ComplexF64}(0)
        hmat = AdaptiveCrossApproximation.HMatrix(
            kernel,
            space,
            tree;
            isnear=allnear,
            spaceordering=ordering,
            scheduler=SerialScheduler(),
            verbose=false,
        )
        reference = [(1 + 0.5im) / (1 + norm(x - y)) for x in space, y in space]
        x = randn(ComplexF64, length(space))
        @test kernel.evaluations == AdaptiveCrossApproximation.nnz(hmat)
        @test kernel.evaluations < length(space)^2
        diagonalentries = sum(length, hmat.nearinteractions.diagonals)
        @test 2 * AdaptiveCrossApproximation.nnz(hmat) == length(space)^2 + diagonalentries
        @test hmat.nearinteractions isa SymmetricBlockMatrix
        @test Matrix(hmat) ≈ reference
        @test hmat * x ≈ reference * x
        @test transpose(hmat) * x ≈ transpose(reference) * x
        @test adjoint(hmat) * x ≈ adjoint(reference) * x
        @test AdaptiveCrossApproximation.nnz(hmat) ==
            sum(length, hmat.nearinteractions.diagonals) +
              sum(length, hmat.nearinteractions.offdiagonals)
    end

    # Self blocks are assembled in full, including when leaves are large.
    cluster = symmetric_test_tree(points; minvalues=length(points) + 1)
    tree = H2Trees.BlockTree(cluster, cluster)
    kernel = CountedSymmetricKernel{ComplexF64}(0)
    nears = AdaptiveCrossApproximation.assemblenears(
        kernel,
        points,
        tree,
        AdaptiveCrossApproximation.PreserveSpaceOrder();
        isnear=allnear,
        verbose=false,
    )
    @test H2Trees.isleaf(cluster, H2Trees.root(cluster))
    @test kernel.evaluations == length(points)^2
    @test Matrix(nears) ≈ dense

    parallelcluster = symmetric_test_tree(points; minvalues=3)
    paralleltree = H2Trees.BlockTree(parallelcluster, parallelcluster)
    parallelnears = AdaptiveCrossApproximation.assemblenears(
        CountedSymmetricKernel{ComplexF64}(0),
        points,
        paralleltree,
        AdaptiveCrossApproximation.PreserveSpaceOrder();
        isnear=allnear,
        scheduler=SerialScheduler(),
        mvpscheduler=DynamicScheduler(),
        verbose=false,
    )
    @test parallelnears * ones(ComplexF64, length(points)) ≈
        dense * ones(ComplexF64, length(points))

    # The convenience call constructs the shared tree from one space.
    auto = AdaptiveCrossApproximation.assemble(
        CountedSymmetricKernel{ComplexF64}(0),
        points;
        treekwargs=(; testminvalues=3, trialminvalues=3),
        isnear=allnear,
        scheduler=SerialScheduler(),
        verbose=false,
    )
    @test Matrix(auto) ≈ dense
    sharedtree = AdaptiveCrossApproximation._tree(
        AdaptiveCrossApproximation.H2Tree(), points; minvalues=3
    )
    @test H2Trees.testtree(sharedtree) === H2Trees.trialtree(sharedtree)
    @test points == [SVector(Float64(i), 0.0) for i in 1:12]

    # Compare the symmetric traversal with the ordinary near field when the
    # admissibility test yields both near and far interactions.
    linepoints = [SVector(Float64(i), 0.0) for i in 1:64]
    linecluster = symmetric_test_tree(linepoints; minvalues=4)
    linetree = H2Trees.BlockTree(linecluster, linecluster)
    nearpredicate = AdaptiveCrossApproximation.isnear(0.5)
    kernel = CountedSymmetricKernel{ComplexF64}(0)
    halfnear = AdaptiveCrossApproximation.assemblenears(
        kernel,
        linepoints,
        linetree,
        AdaptiveCrossApproximation.PreserveSpaceOrder();
        isnear=nearpredicate,
        verbose=false,
    )
    fullnear = AdaptiveCrossApproximation.assemblenears(
        kernel,
        linepoints,
        linepoints,
        linetree,
        AdaptiveCrossApproximation.PreserveSpaceOrder();
        isnear=nearpredicate,
        verbose=false,
    )
    @test Matrix(halfnear) ≈ Matrix(fullnear)
    hmat = AdaptiveCrossApproximation.HMatrix(
        kernel,
        linepoints,
        linetree;
        isnear=nearpredicate,
        tol=1e-5,
        scheduler=SerialScheduler(),
        verbose=false,
    )
    reference = [(1 + 0.5im) / (1 + norm(x - y)) for x in linepoints, y in linepoints]
    @test sum(length(level.offdiagonals) for level in hmat.farinteractions) > 0
    fullhmat = AdaptiveCrossApproximation.HMatrix(
        CountedSymmetricKernel{ComplexF64}(0),
        linepoints,
        linepoints,
        linetree;
        isnear=nearpredicate,
        tol=1e-5,
        scheduler=SerialScheduler(),
        verbose=false,
    )
    @test sum(length(level.blocks) for level in fullhmat.farinteractions) ==
        2 * sum(length(level.offdiagonals) for level in hmat.farinteractions)
    fullfactors = sum(
        length(block.U) + length(block.V) for level in fullhmat.farinteractions for
        block in level.blocks
    )
    halffactors = sum(
        length(block.U) + length(block.V) for level in hmat.farinteractions for
        block in level.offdiagonals
    )
    # General assembly chooses pivots independently in the transposed blocks,
    # so their adaptive ranks need not match exactly.
    @test isapprox(2 * halffactors, fullfactors; rtol=0.05)
    kernel.evaluations = 0
    AdaptiveCrossApproximation.assemblefars(
        kernel,
        linepoints,
        linetree,
        AdaptiveCrossApproximation.PreserveSpaceOrder();
        isnear=nearpredicate,
        compressor=ACA(; tol=1e-5),
        verbose=false,
    )
    halfcalls = kernel.evaluations
    kernel.evaluations = 0
    AdaptiveCrossApproximation.assemblefars(
        kernel,
        linepoints,
        linepoints,
        linetree,
        AdaptiveCrossApproximation.PreserveSpaceOrder();
        isnear=nearpredicate,
        compressor=ACA(; tol=1e-5),
        verbose=false,
    )
    @test halfcalls > 0
    @test isapprox(2 * halfcalls, kernel.evaluations; rtol=0.05)
    @test AdaptiveCrossApproximation.nnz(fullhmat) ==
        sum(length, fullhmat.nearinteractions.blocks) + sum(
        length(block.U) + length(block.V) for level in fullhmat.farinteractions for
        block in level.blocks
    )
    @test AdaptiveCrossApproximation.nnz(hmat) ==
        sum(length, hmat.nearinteractions.diagonals) +
          sum(length, hmat.nearinteractions.offdiagonals) +
          sum(
              length(block.U) + length(block.V) for level in hmat.farinteractions for
              block in level.offdiagonals
          )
    @test Matrix(AdaptiveCrossApproximation.nearmatrix(hmat)) +
          Matrix(AdaptiveCrossApproximation.farmatrix(hmat)) ≈ Matrix(hmat)
    @test norm(Matrix(hmat) - Matrix(fullhmat)) / norm(Matrix(fullhmat)) < 1e-4
    @test norm(Matrix(hmat) - reference) / norm(reference) < 1e-4
    farx = randn(ComplexF64, length(linepoints))
    @test norm(hmat * farx - reference * farx) / norm(reference * farx) < 1e-4
    @test norm(transpose(hmat) * farx - transpose(reference) * farx) /
          norm(transpose(reference) * farx) < 1e-4
    @test norm(adjoint(hmat) * farx - adjoint(reference) * farx) /
          norm(adjoint(reference) * farx) < 1e-4

    parallelhmat = AdaptiveCrossApproximation.HMatrix(
        CountedSymmetricKernel{ComplexF64}(0),
        linepoints,
        linetree;
        isnear=nearpredicate,
        tol=1e-5,
        scheduler=DynamicScheduler(),
        mvpscheduler=DynamicScheduler(),
        verbose=false,
    )
    @test norm(parallelhmat * farx - reference * farx) / norm(reference * farx) < 1e-4

    # Reject unsupported symmetric storage before mutating spaces or tree indices.
    permutedpoints = reverse(copy(linepoints))
    permutedcluster = symmetric_test_tree(permutedpoints; minvalues=4)
    permutedtree = H2Trees.BlockTree(permutedcluster, permutedcluster)
    originalpoints = copy(permutedpoints)
    originalindices = copy(H2Trees.values(permutedcluster, H2Trees.root(permutedcluster)))
    unusedkernel = CountedSymmetricKernel{ComplexF64}(0)
    ordering = AdaptiveCrossApproximation.PermuteSpaceInPlace()
    @test_throws ArgumentError AdaptiveCrossApproximation.HMatrix(
        unusedkernel, permutedpoints, permutedtree; spaceordering=ordering, verbose=false
    )
    @test_throws ArgumentError AdaptiveCrossApproximation.assemble(
        unusedkernel,
        permutedpoints;
        tree=permutedtree,
        spaceordering=ordering,
        verbose=false,
    )
    @test_throws ArgumentError AdaptiveCrossApproximation.assemblenears(
        unusedkernel, permutedpoints, permutedtree, ordering; verbose=false
    )
    @test_throws ArgumentError AdaptiveCrossApproximation.assemblefars(
        unusedkernel, permutedpoints, permutedtree, ordering; verbose=false
    )
    @test permutedpoints == originalpoints
    @test H2Trees.values(permutedcluster, H2Trees.root(permutedcluster)) == originalindices
    @test unusedkernel.evaluations == 0

    # An admissibility predicate may leave the near field empty.
    emptynears = AdaptiveCrossApproximation.assemblenears(
        kernel,
        points,
        tree,
        AdaptiveCrossApproximation.PreserveSpaceOrder();
        isnear=(treea, treeb, a, b) -> false,
        mvpscheduler=DynamicScheduler(),
        verbose=false,
    )
    @test Matrix(emptynears) == zeros(ComplexF64, length(points), length(points))

    # A custom predicate can classify the root self block as far. It is
    # compressed once and stored as one symmetrized diagonal block.
    farself = AdaptiveCrossApproximation.HMatrix(
        CountedSymmetricKernel{ComplexF64}(0),
        points,
        tree;
        isnear=(treea, treeb, a, b) -> false,
        tol=1e-9,
        maxrank=length(points) + 1,
        scheduler=SerialScheduler(),
        verbose=false,
    )
    @test length(farself.farinteractions) == 1
    @test norm(Matrix(farself) - dense) / norm(dense) < 1e-4
    @test farself * ones(ComplexF64, length(points)) ≈
        Matrix(farself) * ones(ComplexF64, length(points))

    # A custom predicate can leave off-diagonal near pairs with no self block.
    offonly = AdaptiveCrossApproximation.assemblenears(
        kernel,
        points,
        paralleltree,
        AdaptiveCrossApproximation.PreserveSpaceOrder();
        isnear=(treea, treeb, a, b) -> a != b || !H2Trees.isleaf(treea, a),
        mvpscheduler=DynamicScheduler(),
        verbose=false,
    )
    @test offonly isa SymmetricBlockMatrix
    @test diag(Matrix(offonly)) == zeros(ComplexF64, length(points))
    @test offonly * ones(ComplexF64, length(points)) ≈
        Matrix(offonly) * ones(ComplexF64, length(points))
    serialoffonly = AdaptiveCrossApproximation.assemblenears(
        kernel,
        points,
        paralleltree,
        AdaptiveCrossApproximation.PreserveSpaceOrder();
        isnear=(treea, treeb, a, b) -> a != b || !H2Trees.isleaf(treea, a),
        mvpscheduler=SerialScheduler(),
        verbose=false,
    )
    @test isempty(serialoffonly.diagonals)
    @test serialoffonly * ones(ComplexF64, length(points)) ≈
        Matrix(serialoffonly) * ones(ComplexF64, length(points))

    # Match test and trial quadrature orders so the dense reference is symmetric
    # to the accuracy of the singular-pair rule.
    mesh = CompScienceMeshes.meshsphere(1.0, 1.2)
    space = BEAST.raviartthomas(mesh)
    op = BEAST.Maxwell3D.singlelayer(; wavenumber=1.0)
    quadstrat = BEAST.DoubleNumSauterQstrat(8, 8, 0, 8, 8, 8)
    cluster = symmetric_test_tree(BEAST.positions(space); minvalues=length(space))
    tree = H2Trees.BlockTree(cluster, cluster)
    near = AdaptiveCrossApproximation.HMatrix(
        op,
        space,
        tree;
        isnear=allnear,
        nearmatrixdata=quadstrat,
        scheduler=SerialScheduler(),
        verbose=false,
    )
    assembled = BEAST.assemble(op, space, space; quadstrat=quadstrat)
    fullnear = AdaptiveCrossApproximation.assemblenears(
        op,
        space,
        space,
        tree,
        AdaptiveCrossApproximation.PreserveSpaceOrder();
        isnear=allnear,
        matrixdata=quadstrat,
        scheduler=SerialScheduler(),
        verbose=false,
    )
    @test Matrix(fullnear) == Matrix(assembled)
    @test norm(assembled - transpose(assembled)) / norm(assembled) < 1e-7
    @test norm(triu(Matrix(near) - assembled)) / norm(assembled) < 1e-9
    @test norm(Matrix(near) - assembled) / norm(assembled) < 1e-7
    x = ones(ComplexF64, length(space))
    @test norm(near * x - assembled * x) / norm(assembled * x) < 1e-7
    mixedcluster = symmetric_test_tree(BEAST.positions(space); minvalues=8)
    mixedtree = H2Trees.BlockTree(mixedcluster, mixedcluster)
    # Off-diagonal near columns are combined during traversal. Assemble these
    # final-sized rectangles and the diagonal blocks directly on the CPU.
    diagindices, rowindices, colindices = AdaptiveCrossApproximation.symmetricnearinteractions(
        mixedtree; isnear=allnear
    )
    for scheduler in (SerialScheduler(), DynamicScheduler())
        groupednear = AdaptiveCrossApproximation.assemblenears(
            op,
            space,
            mixedtree,
            AdaptiveCrossApproximation.PreserveSpaceOrder();
            isnear=allnear,
            matrixdata=quadstrat,
            scheduler=scheduler,
            mvpscheduler=scheduler,
            verbose=false,
        )
        @test length(groupednear.offdiagonals) == length(rowindices)
        @test all(
            size(block) == (length(rows), length(cols)) for
            (block, rows, cols) in zip(groupednear.offdiagonals, rowindices, colindices)
        )
        @test sum(length, groupednear.diagonals) ==
            sum(length(ids)^2 for ids in diagindices)
        @test norm(Matrix(groupednear) - assembled) / norm(assembled) < 1e-7
        @test norm(groupednear * x - assembled * x) / norm(assembled * x) < 1e-7
    end
    mixed = AdaptiveCrossApproximation.HMatrix(
        op,
        space,
        mixedtree;
        isnear=AdaptiveCrossApproximation.isnear(2.0),
        nearmatrixdata=quadstrat,
        farmatrixdata=quadstrat,
        tol=1e-4,
        scheduler=SerialScheduler(),
        verbose=false,
    )
    @test sum(length(level.offdiagonals) for level in mixed.farinteractions) > 0
    @test norm(Matrix(mixed) - assembled) / norm(assembled) < 1e-4
    @test norm(mixed * x - assembled * x) / norm(assembled * x) < 1e-4
    kmeanstree = AdaptiveCrossApproximation._tree(
        AdaptiveCrossApproximation.KMeansTreeBackend(), space; minvalues=8
    )
    @test H2Trees.testtree(kmeanstree) === H2Trees.trialtree(kmeanstree)
    autobeast = AdaptiveCrossApproximation.assemble(
        op,
        space;
        treekwargs=(; minvalues=8),
        quadstrat=quadstrat,
        tol=1e-4,
        scheduler=SerialScheduler(),
        verbose=false,
    )
    @test norm(Matrix(autobeast) - assembled) / norm(assembled) < 1e-4
    @test Matrix(AdaptiveCrossApproximation.assemble(BEAST.Identity(), space)) ≈
        Matrix(BEAST.assemble(BEAST.Identity(), space, space))
end

@testset "Symmetric H-matrix scalar types and schedulers" begin
    points = [SVector(Float64(i), 0.0) for i in 1:32]
    cluster = symmetric_test_tree(points; minvalues=4)
    tree = H2Trees.BlockTree(cluster, cluster)
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        scale = T <: Real ? T(1) : T(1 + 0.5im)
        kernel = TypedSymmetricKernel(scale)
        reference = [kernel(x, y) for x in points, y in points]
        x = ones(T, length(points))
        for scheduler in (SerialScheduler(), DynamicScheduler())
            hmat = AdaptiveCrossApproximation.HMatrix(
                kernel,
                points,
                tree;
                isnear=AdaptiveCrossApproximation.isnear(1.0),
                tol=1e-5,
                scheduler=scheduler,
                verbose=false,
            )
            @test eltype(hmat) == T
            @test norm(Matrix(hmat) - reference) / norm(reference) < 1e-4
            @test norm(hmat * x - reference * x) / norm(reference * x) < 1e-4
            @test transpose(hmat) * x ≈ hmat * x
            @test adjoint(hmat) * x ≈ conj.(hmat * x)
        end
    end
end
