using AdaptiveCrossApproximation
using H2Trees
using LinearAlgebra
using StaticArrays
using Test

# Exact blocks bypass ACA entirely. Cluster frontiers deliberately include
# internal nodes, and the point orders differ on the two axes.
function arithmetic_workspace_fixture(ordering; symmetric=false)
    ACA = AdaptiveCrossApproximation
    rowpoints = [
        SVector(Float64(i), 0.0) for
        i in [11, 2, 15, 4, 9, 6, 13, 8, 1, 14, 3, 16, 5, 12, 7, 10]
    ]
    colpoints = if symmetric
        rowpoints
    else
        [SVector(Float64(i), 0.0) for i in [6, 11, 2, 9, 4, 12, 1, 8, 3, 10, 5, 7]]
    end
    builder = H2Trees.TwoNTreeBuilder(; minhalfsize=0.1, minvalues=2)
    rowtree = H2Trees.buildtree(rowpoints; builder)
    coltree = symmetric ? rowtree : H2Trees.buildtree(colpoints; builder)
    tree = H2Trees.BlockTree(rowtree, coltree)
    ordering(tree, rowpoints, colpoints)

    # Obtain the expected global order independently of ACA.permutation.
    rowperm = reduce(
        vcat, [copy(H2Trees.values(rowtree, n)) for n in H2Trees.leaves(rowtree)]
    )
    colperm = reduce(
        vcat, [copy(H2Trees.values(coltree, n)) for n in H2Trees.leaves(coltree)]
    )
    rownodes = [
        n for child in H2Trees.children(rowtree, H2Trees.root(rowtree)) for
        n in H2Trees.children(rowtree, child)
    ]
    colnodes =
        symmetric ? rownodes : collect(H2Trees.children(coltree, H2Trees.root(coltree)))
    left = ComplexF64[complex(x[1], k) for x in rowpoints, k in 1:2]
    right = if symmetric
        Matrix(transpose(left))
    else
        ComplexF64[complex(k + y[1], k - y[1]) for k in 1:2, y in colpoints]
    end
    reference = left * right
    near = Matrix{ComplexF64}[]
    far = ACA.LowRankMatrix{ComplexF64}[]
    rows, cols, storageidx = Int[], Int[], Int[]
    farflags, transposed = BitVector(), BitVector()
    for (i, rownode) in enumerate(rownodes), (j, colnode) in enumerate(colnodes)
        symmetric && i > j && continue
        rowindices = H2Trees.values(rowtree, rownode)
        colindices = H2Trees.values(coltree, colnode)
        isfar = symmetric ? j > i + 1 : isodd(i + j)
        if isfar
            push!(far, ACA.LowRankMatrix(left[rowindices, :], right[:, colindices]))
        else
            block = copy(reference[rowindices, colindices])
            for b in axes(block, 2), a in axes(block, 1)
                block[a, b] +=
                    symmetric && i == j ? complex(a + b, a * b) : complex(a + 2b, a - b)
            end
            push!(near, block)
            reference[rowindices, colindices] = block
            if symmetric && i != j
                reference[colindices, rowindices] = transpose(block)
            end
        end
        storage = isfar ? length(far) : length(near)
        push!(rows, rownode)
        push!(cols, colnode)
        push!(farflags, isfar)
        push!(transposed, false)
        push!(storageidx, storage)
        if symmetric && i != j
            push!(rows, colnode)
            push!(cols, rownode)
            push!(farflags, isfar)
            push!(transposed, true)
            push!(storageidx, storage)
        end
    end
    # Physical blocks stay in construction order; CSR must use its storage map.
    csr, csc = ACA._blockindices(
        rows,
        cols,
        farflags,
        transposed,
        storageidx,
        H2Trees.numberofnodes(rowtree),
        H2Trees.numberofnodes(coltree),
    )
    core = ACA.HMatrixCore{ComplexF64}(
        tree, near, far, csr, csc, ordering, size(reference), 1e-10
    )
    return (; core, reference, rowperm, colperm)
end

# Derive contiguous tree positions from external cluster indices. This avoids
# assuming that original indices are ranges or using ACA's clusterindices helper.
function workspace_test_ranges(tree, perm)
    positions = invperm(perm)
    return map(1:H2Trees.numberofnodes(tree)) do node
        indices = positions[H2Trees.values(tree, node)]
        @test indices == collect(first(indices):last(indices))
        return first(indices):last(indices)
    end
end

function workspace_test_dense(workspace, rowranges, colranges; traversal=:csr)
    csr, csc = workspace.csr, workspace.csc
    matrix = zeros(eltype(workspace), size(workspace))
    coverage = zeros(Int, size(workspace))
    validsizes = true
    entries = Tuple{Int,Int,Int}[]
    if traversal === :csr
        for row in 1:(length(csr.rowptr) - 1)
            for k in csr.rowptr[row]:(csr.rowptr[row + 1] - 1)
                push!(entries, (row, csr.colnodes[k], k))
            end
        end
    else
        for col in 1:(length(csc.colptr) - 1)
            for k in csc.colptr[col]:(csc.colptr[col + 1] - 1)
                push!(entries, (csc.rownodes[k], col, csc.csrmap[k]))
            end
        end
    end
    for (row, col, k) in entries
        storage = csr.storageidx[k]
        block = if csr.isfar[k]
            factors = workspace.far[storage]
            factors.U * factors.V
        else
            workspace.near[storage]
        end
        logical = csr.transposed[k] ? transpose(block) : block
        rows, cols = rowranges[row], colranges[col]
        validsizes &= size(logical) == (length(rows), length(cols))
        matrix[rows, cols] = logical
        coverage[rows, cols] .+= 1
    end
    @test validsizes
    @test all(==(1), coverage)
    return matrix
end

@testset "Tree index and matrix-vector products" begin
    ACA = AdaptiveCrossApproximation
    for symmetric in (false, true),
        ordering in (ACA.PreserveSpaceOrder(), ACA.PermuteSpaceInPlace())

        @testset "symmetric=$symmetric, ordering=$(typeof(ordering))" begin
            (; core, reference, rowperm, colperm) = arithmetic_workspace_fixture(
                ordering; symmetric
            )
            rowtree, coltree = H2Trees.testtree(core.tree), H2Trees.trialtree(core.tree)
            leavesbefore = (
                [copy(H2Trees.values(rowtree, n)) for n in H2Trees.leaves(rowtree)],
                [copy(H2Trees.values(coltree, n)) for n in H2Trees.leaves(coltree)],
            )
            index = @inferred ACA._TreeIndex(core)

            @testset "Permutations and tree-order ranges" begin
                @test sort(rowperm) == collect(1:size(core, 1))
                @test sort(colperm) == collect(1:size(core, 2))
                if ordering isa ACA.PermuteSpaceInPlace
                    @test index.rowperm isa ACA.IdentityPermutation
                    @test index.colperm isa ACA.IdentityPermutation
                    @test rowperm == collect(1:size(core, 1))
                    @test colperm == collect(1:size(core, 2))
                else
                    @test index.rowperm == rowperm
                    @test index.colperm == colperm
                    @test rowperm != collect(1:size(core, 1))
                    @test colperm != collect(1:size(core, 2))
                end
                if symmetric
                    @test index.rowperm === index.colperm
                else
                    @test rowtree !== coltree
                    @test size(core, 1) != size(core, 2)
                    @test rowperm != colperm
                end
                @test leavesbefore == (
                    [H2Trees.values(rowtree, n) for n in H2Trees.leaves(rowtree)],
                    [H2Trees.values(coltree, n) for n in H2Trees.leaves(coltree)],
                )
                rowranges = workspace_test_ranges(rowtree, rowperm)
                colranges = workspace_test_ranges(coltree, colperm)
                @test index.rowranges == rowranges
                @test index.colranges == colranges
                expected = reference[rowperm, colperm]
                @test workspace_test_dense(core, rowranges, colranges) ≈ expected
                @test workspace_test_dense(core, rowranges, colranges; traversal=:csc) ≈
                    expected
                @test Matrix(core) ≈ reference
            end

            @testset "Matrix-vector products" begin
                x = ComplexF64[complex(i, -i / 3) for i in 1:size(core, 2)]
                xt = ComplexF64[complex(-i / 2, i) for i in 1:size(core, 1)]
                X = reshape(ComplexF64.(1:(3 * size(core, 2))), size(core, 2), 3)
                for ntasks in (1, 2)
                    A = ACA.MatVecWorkspace(core; ntasks)
                    @test A * x ≈ reference * x
                    @test transpose(A) * xt ≈ transpose(reference) * xt
                    @test A' * xt ≈ reference' * xt
                    @test A * X ≈ reference * X
                    y = ones(ComplexF64, size(core, 1))
                    @test mul!(y, A, x, 2, -1) ≈ 2 * reference * x .- 1
                    B = ACA.MatVecWorkspace(A)
                    @test B.buffers !== A.buffers
                    @test B * x ≈ reference * x
                end
            end
        end
    end
end
