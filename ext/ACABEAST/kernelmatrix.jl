function AdaptiveCrossApproximation.AbstractKernelMatrix(
    operator::BEAST.IntegralOperator,
    testspace::BEAST.Space,
    trialspace::BEAST.Space;
    matrixdata=BEAST.defaultquadstrat(operator, testspace, trialspace),
)
    return AdaptiveCrossApproximation.beastkernelmatrix(
        operator, testspace, trialspace, matrixdata
    )
end

function AdaptiveCrossApproximation.beastkernelmatrix(
    operator::BEAST.IntegralOperator,
    testspace::BEAST.Space,
    trialspace::BEAST.Space,
    quadstrat,
)
    assembler = BEAST.blockassembler(operator, testspace, trialspace; quadstrat)

    return AdaptiveCrossApproximation.BEASTKernelMatrix{scalartype(operator)}(assembler)
end

struct BlockStoreFunctor{M}
    matrix::M
end

function (f::BlockStoreFunctor)(v, m, n)
    @views f.matrix[m, n] += v
    return nothing
end

function (blk::AdaptiveCrossApproximation.BEASTKernelMatrix)(matrixblock, tdata, sdata)
    blk.nearassembler(tdata, sdata, BlockStoreFunctor(matrixblock))
    return nothing
end

struct LocalBEASTKernelMatrix{T,M,R,C} <: AdaptiveCrossApproximation.AbstractKernelMatrix{T}
    matrix::M
    rowdata::R
    columndata::C
end

function Base.size(matrix::LocalBEASTKernelMatrix, dim=nothing)
    return size(matrix.matrix, dim)
end

function AdaptiveCrossApproximation.localkernelmatrix(
    matrix::AdaptiveCrossApproximation.BEASTKernelMatrix, rows, columns
)
    return _localkernelmatrix(matrix, rows, columns, matrix.nearassembler.quadstrat)
end

_localkernelmatrix(matrix, rows, columns, quadstrat) = matrix

function _localkernelmatrix(
    matrix::AdaptiveCrossApproximation.BEASTKernelMatrix,
    rows,
    columns,
    quadstrat::BEAST.DoubleNumQStrat,
)
    return LocalBEASTKernelMatrix(matrix, rows, columns, quadstrat)
end

# BEAST's CPU assembler computes these ids inline, so keep a local copy here.
function _active_element_ids(space, ids)
    return unique!(
        sort!(collect(Int, shape.cellid for id in ids for shape in space.fns[id]))
    )
end

function LocalBEASTKernelMatrix(
    matrix::AdaptiveCrossApproximation.BEASTKernelMatrix{T},
    rows,
    columns,
    ::BEAST.DoubleNumQStrat,
) where {T}
    assembler = matrix.nearassembler
    testspace = assembler.tfs
    trialspace = assembler.bfs
    rowtestspace = BEAST.subset(testspace, first(rows):first(rows))
    columntrialspace = BEAST.subset(trialspace, first(columns):first(columns))

    testelementids = _active_element_ids(assembler.tfs, rows)
    trialelementids = _active_element_ids(assembler.bfs, columns)
    testelements = view(assembler.testelements, testelementids)
    trialelements = view(assembler.trialelements, trialelementids)
    testassemblydata = BEAST.reduce_assembly_data(
        assembler.testassemblydata, rows, testelementids
    )
    trialassemblydata = BEAST.reduce_assembly_data(
        assembler.trialassemblydata, columns, trialelementids
    )
    testshapes = BEAST.refspace(testspace)
    trialshapes = BEAST.refspace(trialspace)

    rowquadraturedata = (
        assembler.quadraturedata[1], view(assembler.quadraturedata[2], :, trialelementids)
    )
    columnquadraturedata = (
        view(assembler.quadraturedata[1], :, testelementids), assembler.quadraturedata[2]
    )
    zlocal = zeros(
        T,
        size(assembler.testassemblydata.data, 2),
        size(assembler.trialassemblydata.data, 2),
    )

    rowdata = (;
        testspace=rowtestspace,
        testshapes,
        trialspace,
        trialelements,
        trialassemblydata,
        trialshapes,
        quadraturedata=rowquadraturedata,
        zlocal=copy(zlocal),
    )
    columndata = (;
        testspace,
        testelements,
        testassemblydata,
        testshapes,
        trialspace=columntrialspace,
        trialshapes,
        quadraturedata=columnquadraturedata,
        zlocal,
    )
    return LocalBEASTKernelMatrix{T,typeof(matrix),typeof(rowdata),typeof(columndata)}(
        matrix, rowdata, columndata
    )
end

function (matrix::LocalBEASTKernelMatrix)(
    matrixblock, row::Integer, columns::AbstractVector{<:Integer}
)
    assembler = matrix.matrix.nearassembler
    store = BlockStoreFunctor(matrixblock)
    data = matrix.rowdata
    testspace = data.testspace
    testspace.fns[1] = assembler.tfs.fns[row]
    testspace.pos[1] = assembler.tfs.pos[row]
    BEAST.assemblerow_body!(
        assembler.biop,
        testspace,
        assembler.testelements,
        data.testshapes,
        data.trialassemblydata,
        data.trialspace,
        data.trialelements,
        data.trialshapes,
        data.zlocal,
        data.quadraturedata,
        store;
        quadstrat=assembler.quadstrat,
    )
    return nothing
end

function (matrix::LocalBEASTKernelMatrix)(
    matrixblock, rows::AbstractVector{<:Integer}, column::Integer
)
    assembler = matrix.matrix.nearassembler
    store = BlockStoreFunctor(matrixblock)
    data = matrix.columndata
    trialspace = data.trialspace
    trialspace.fns[1] = assembler.bfs.fns[column]
    trialspace.pos[1] = assembler.bfs.pos[column]
    BEAST.assemblecol_body!(
        assembler.biop,
        data.testassemblydata,
        data.testspace,
        data.testelements,
        data.testshapes,
        trialspace,
        assembler.trialelements,
        data.trialshapes,
        data.zlocal,
        data.quadraturedata,
        store;
        quadstrat=assembler.quadstrat,
    )
    return nothing
end

function (matrix::LocalBEASTKernelMatrix)(matrixblock, row::Integer, column::Integer)
    return matrix.matrix(matrixblock, row:row, column:column)
end

function (matrix::LocalBEASTKernelMatrix)(matrixblock, rows, columns)
    return matrix.matrix(matrixblock, rows, columns)
end

function AdaptiveCrossApproximation.nextrc!(
    matrixblock, matrix::LocalBEASTKernelMatrix, rows, columns
)
    return matrix(matrixblock, rows, columns)
end
