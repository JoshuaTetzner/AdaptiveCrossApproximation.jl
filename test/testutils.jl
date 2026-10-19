using AdaptiveCrossApproximation
using H2Trees
using LinearAlgebra
using Random
using StaticArrays
using Test

# Shared exact fixtures and independent dense references. No test file includes
# another test file; runtests.jl loads these helpers once before the suites.

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

    # Obtain the expected global order independently of ACA's tree interface.
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
# assuming that original indices are ranges or using ACA's tree interface.
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

# Reuse the exact, irregular-tree fixture from the tree-index tests with full storage.
function hierarchical_core(T; ordering=AdaptiveCrossApproximation.PreserveSpaceOrder())
    fixture = arithmetic_workspace_fixture(ordering)
    core = fixture.core
    convertmatrix(A) = T <: Real ? Matrix{T}(real.(A)) : Matrix{T}(A)
    near = map(convertmatrix, core.near)
    far = [
        AdaptiveCrossApproximation.LowRankMatrix{T}(convertmatrix(R.U), convertmatrix(R.V))
        for R in core.far
    ]
    return AdaptiveCrossApproximation.HMatrixCore{T}(
        core.tree, near, far, core.csr, core.csc, core.ordering, size(core), core.tol
    )
end

# Tree-order view with its dense tree-order reference, built from the blocks.
function hierarchical_fixture(T; kwargs...)
    W = AdaptiveCrossApproximation._TreeView(hierarchical_core(T; kwargs...))
    D = workspace_test_dense(W, W.rowranges, W.colranges)
    return W, D
end

# The same logical matrix on the same tree, optionally with reversed physical
# storage order.
function hierarchical_source(W; reversed=true)
    ACA = AdaptiveCrossApproximation
    csr = W.csr
    near = [copy(D) for D in W.near]
    far = [ACA.LowRankMatrix(copy(R.U), copy(R.V)) for R in W.far]
    indices = copy(csr.storageidx)
    if reversed
        reverse!(near)
        reverse!(far)
        for k in eachindex(indices)
            indices[k] = (csr.isfar[k] ? length(far) : length(near)) + 1 - indices[k]
        end
    end
    index = ACA.BlockCSR(
        copy(csr.rowptr), copy(csr.colnodes), copy(csr.isfar), copy(csr.transposed), indices
    )
    core = ACA.HMatrixCore{eltype(W)}(
        W.tree, near, far, index, W.csc, ACA.PreserveSpaceOrder(), size(W), 1e-10
    )
    return ACA._TreeView(core)
end

function hierarchical_thin_bytes(Y, H, X, scratch, side)
    if side === :left
        AdaptiveCrossApproximation._thinleft!(Y, H, X, 0.3, 0, scratch)
        return @allocated AdaptiveCrossApproximation._thinleft!(Y, H, X, 0.3, 0, scratch)
    end
    AdaptiveCrossApproximation._thinright!(Y, X, H, 0.3, 0, scratch)
    return @allocated AdaptiveCrossApproximation._thinright!(Y, X, H, 0.3, 0, scratch)
end

hierarchical_add!(C, A, alpha, scratch, options) =
    AdaptiveCrossApproximation._hierarchicaladd!(
        C, A, alpha, [AdaptiveCrossApproximation._Worker(scratch)], options
    )

# Independently assembled exact blocks; no ACA or new product kernel is used
# for reference construction. Axes/frontiers and dense/far classifications vary.
function product_test_fixture(
    rng, T, rowtree, coltree; rowfront=:children, colfront=:leaves, kind=:mixed
)
    ACA = AdaptiveCrossApproximation
    frontier(tree, choice) =
        if choice === :root
            [H2Trees.root(tree)]
        elseif choice === :leaves
            collect(H2Trees.leaves(tree))
        else
            collect(H2Trees.children(tree, H2Trees.root(tree)))
        end
    rr, cc = ACA._clusterranges(rowtree), ACA._clusterranges(coltree)
    m, n = length(rr[H2Trees.root(rowtree)]), length(cc[H2Trees.root(coltree)])
    left, right = randn(rng, T, m, 2), randn(rng, T, 2, n)
    near, far = Matrix{T}[], ACA.LowRankMatrix{T}[]
    rows, cols, indices, flags = Int[], Int[], Int[], BitVector()
    reference = zeros(T, m, n)
    for (i, row) in enumerate(frontier(rowtree, rowfront)),
        (j, col) in enumerate(frontier(coltree, colfront))

        isfar = kind === :far || kind === :mixed && isodd(i + j)
        U, V = left[rr[row], :], right[:, cc[col]]
        if isfar
            reference[rr[row], cc[col]] = U * V
            push!(far, ACA.LowRankMatrix(U, V))
        else
            D = U * V + randn(rng, T, length(rr[row]), length(cc[col]))
            reference[rr[row], cc[col]] = D
            push!(near, D)
        end
        push!(rows, row)
        push!(cols, col)
        push!(indices, isfar ? length(far) : length(near))
        push!(flags, isfar)
    end
    csr, csc = ACA._blockindices(
        rows,
        cols,
        flags,
        falses(length(flags)),
        indices,
        H2Trees.numberofnodes(rowtree),
        H2Trees.numberofnodes(coltree),
    )
    core = ACA.HMatrixCore{T}(
        H2Trees.BlockTree(rowtree, coltree),
        near,
        far,
        csr,
        csc,
        ACA.PreserveSpaceOrder(),
        (m, n),
        1e-10,
    )
    return ACA._TreeView(core), reference
end

product_test_dense(W) = workspace_test_dense(W, W.rowranges, W.colranges)

product_test_block(H::AdaptiveCrossApproximation._TreeView) =
    AdaptiveCrossApproximation.blockview(H)
product_test_block(H) = H

function product_test_plan(C, A, B; kwargs...)
    return AdaptiveCrossApproximation._ProductPlan(
        product_test_block(C), product_test_block(A), product_test_block(B); kwargs...
    )
end

function product_test_addmul!(P, alpha, scratch; atol=0, rtol=1e-6, maxrank=typemax(Int))
    ACA = AdaptiveCrossApproximation
    workers = [ACA._Worker(scratch)]
    return ACA._addmul!(P, alpha, workers, ACA._productgraph(P), (; atol, rtol, maxrank))
end

function product_test_addmul!(C, A, B, alpha, scratch; batchrank=128, kwargs...)
    return product_test_addmul!(
        product_test_plan(C, A, B; batchrank), alpha, scratch; kwargs...
    )
end

product_test_arrays(scratch) =
    (scratch.productleft, scratch.productright, scratch.aggregate)

# Exact factors and independent dense triangular references; no ACA or
# hierarchical LU is used to manufacture a supposedly correct test answer.
function triangular_test_fixture(
    rng,
    T;
    coarsened=false,
    n=29,
    multidimensional=false,
    rank=3,
    ordering=AdaptiveCrossApproximation.PreserveSpaceOrder(),
    factorize=true,
)
    ACA = AdaptiveCrossApproximation
    points = [
        if multidimensional
            SVector(Float64(mod(i, 7)), Float64(i ÷ 7))
        else
            SVector(Float64(i)^2, 0.0)
        end for i in randperm(rng, n)
    ]
    cluster = H2Trees.buildtree(
        points; builder=H2Trees.TwoNTreeBuilder(; minvalues=3, minhalfsize=0.001)
    )
    tree = H2Trees.BlockTree(cluster, cluster)
    ordering(tree, points, points)
    ranges = ACA._clusterranges(cluster)
    root = H2Trees.root(cluster)
    coarse = coarsened ? first(H2Trees.children(cluster, root)) : 0
    near, far = Matrix{T}[], ACA.LowRankMatrix{T}[]
    rows, cols, indices = Int[], Int[], Int[]
    flags = BitVector()
    function terminal(row, col)
        rr, cc = ranges[row], ranges[col]
        diagonal = row == col
        isfar = !diagonal && isodd(row + col)
        stop =
            isfar ||
            diagonal && row == coarse ||
            H2Trees.isleaf(cluster, row) ||
            H2Trees.isleaf(cluster, col)
        if !stop
            for r in H2Trees.children(cluster, row), c in H2Trees.children(cluster, col)
                terminal(r, c)
            end
            return nothing
        end
        if isfar
            U, V = randn(rng, T, length(rr), rank), randn(rng, T, rank, length(cc))
            push!(far, ACA.LowRankMatrix(U, V))
        else
            D = randn(rng, T, length(rr), length(cc))
            if diagonal
                D += 10length(rr)*I
                D = D[circshift(collect(1:length(rr)), 1), :]
            end
            push!(near, D)
        end
        index = isfar ? length(far) : length(near)
        push!(rows, row)
        push!(cols, col)
        push!(indices, index)
        push!(flags, isfar)
        return nothing
    end
    terminal(root, root)
    csr, csc = ACA._blockindices(
        rows,
        cols,
        flags,
        falses(length(flags)),
        indices,
        H2Trees.numberofnodes(cluster),
        H2Trees.numberofnodes(cluster),
    )
    core = ACA.HMatrixCore{T}(tree, near, far, csr, csc, ordering, (n, n), 1e-10)
    factorize || return core, nothing
    W = ACA._TreeView(core)
    leaves = [
        ACA.leaflu!(W.near[csr.storageidx[k]]; node=W.entryrows[k]) for
        k in eachindex(csr.storageidx) if W.entryrows[k] == csr.colnodes[k]
    ]
    return W, leaves
end

function triangular_test_dense(W)
    L, U = zeros(eltype(W), size(W)), zeros(eltype(W), size(W))
    csr = W.csr
    for k in eachindex(csr.storageidx)
        row, col = W.entryrows[k], csr.colnodes[k]
        rr, cc = W.rowranges[row], W.colranges[col]
        logical =
            csr.isfar[k] ? Matrix(W.far[csr.storageidx[k]]) : W.near[csr.storageidx[k]]
        if row == col
            L[rr, cc] = UnitLowerTriangular(logical)
            U[rr, cc] = UpperTriangular(logical)
        elseif first(rr) > first(cc)
            L[rr, cc] = logical
        else
            U[rr, cc] = logical
        end
    end
    return L, U
end

# Transposed factors are a solve option of the internal kernel.
struct TriangularTestTranspose{H}
    parent::H
end
triangular_test_op(H, op) = op === :N ? H : TriangularTestTranspose(H)
triangular_test_parts(H::AdaptiveCrossApproximation.HMatrixTriangular) = (H, Val(:N))
triangular_test_parts(H::TriangularTestTranspose) = (H.parent, Val(:T))

function triangular_test_solve!(B, H, scratch, side)
    F, op = triangular_test_parts(H)
    if side === :left
        return AdaptiveCrossApproximation._trisolve!(B, F, scratch, op, Val(:left))
    end
    return AdaptiveCrossApproximation._trisolve!(B, F, scratch, op, Val(:right))
end
function triangular_test_bytes(B, H, scratch, side)
    triangular_test_solve!(B, H, scratch, side)
    return @allocated triangular_test_solve!(B, H, scratch, side)
end
function triangular_test_pivotbytes(B, H)
    AdaptiveCrossApproximation._pivot!(B, H)
    return @allocated AdaptiveCrossApproximation._pivot!(B, H)
end

# Point kernels used by HMatrix and ordering tests.
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

# CUR reconstruction error of block `i` from selected `rank`/pivots.
function _cur_relerr(A, rank, rowpivots, colpivots)
    rank == 0 && return norm(A)  # a nonzero block picked no pivots => full error
    rp = rowpivots[1:rank]
    cp = colpivots[1:rank]
    recon = A[:, cp] * (A[rp, cp] \ A[rp, :])
    return norm(recon - A) / norm(A)
end
