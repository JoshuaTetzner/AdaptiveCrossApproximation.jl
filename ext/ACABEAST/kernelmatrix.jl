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

function _localzlocal(::Type{T}, assembler) where {T}
    return zeros(
        T,
        size(assembler.testassemblydata.data, 2),
        size(assembler.trialassemblydata.data, 2),
    )
end

# Data for sampling rows against the fixed `columns`; `nothing` if no columns are given.
_localrowdata(::Type, assembler, ::Nothing) = nothing

function _localrowdata(::Type{T}, assembler, columns) where {T}
    testspace = assembler.tfs
    trialspace = assembler.bfs
    trialelementids = _active_element_ids(trialspace, columns)
    return (;
        testspace=BEAST.subset(testspace, 1:1),
        testshapes=BEAST.refspace(testspace),
        trialspace,
        trialelements=view(assembler.trialelements, trialelementids),
        trialassemblydata=BEAST.reduce_assembly_data(
            assembler.trialassemblydata, columns, trialelementids
        ),
        trialshapes=BEAST.refspace(trialspace),
        quadraturedata=(
            assembler.quadraturedata[1],
            view(assembler.quadraturedata[2], :, trialelementids),
        ),
        zlocal=_localzlocal(T, assembler),
    )
end

# Data for sampling columns against the fixed `rows`; `nothing` if no rows are given.
_localcolumndata(::Type, assembler, ::Nothing) = nothing

function _localcolumndata(::Type{T}, assembler, rows) where {T}
    testspace = assembler.tfs
    trialspace = assembler.bfs
    testelementids = _active_element_ids(testspace, rows)
    return (;
        testspace,
        testelements=view(assembler.testelements, testelementids),
        testassemblydata=BEAST.reduce_assembly_data(
            assembler.testassemblydata, rows, testelementids
        ),
        testshapes=BEAST.refspace(testspace),
        trialspace=BEAST.subset(trialspace, 1:1),
        trialshapes=BEAST.refspace(trialspace),
        quadraturedata=(
            view(assembler.quadraturedata[1], :, testelementids),
            assembler.quadraturedata[2],
        ),
        zlocal=_localzlocal(T, assembler),
    )
end

function LocalBEASTKernelMatrix(
    matrix::AdaptiveCrossApproximation.BEASTKernelMatrix{T},
    rows,
    columns,
    ::BEAST.DoubleNumQStrat,
) where {T}
    assembler = matrix.nearassembler
    rowdata = _localrowdata(T, assembler, columns)
    columndata = _localcolumndata(T, assembler, rows)
    return LocalBEASTKernelMatrix{T,typeof(matrix),typeof(rowdata),typeof(columndata)}(
        matrix, rowdata, columndata
    )
end

function (matrix::LocalBEASTKernelMatrix)(
    matrixblock, row::Integer, columns::AbstractVector{<:Integer}
)
    assembler = matrix.matrix.nearassembler
    data = matrix.rowdata
    data === nothing && return matrix.matrix(matrixblock, row:row, columns)
    store = BlockStoreFunctor(matrixblock)
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
    data = matrix.columndata
    data === nothing && return matrix.matrix(matrixblock, rows, column:column)
    store = BlockStoreFunctor(matrixblock)
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
