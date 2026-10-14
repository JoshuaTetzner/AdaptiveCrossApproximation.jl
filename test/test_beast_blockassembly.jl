using AdaptiveCrossApproximation
using BEAST
using CompScienceMeshes
using OhMyThreads
using Test

struct CountedBlockAssembler{A}
    assembler::A
    calls::Threads.Atomic{Int}
end

function (assembler::CountedBlockAssembler)(rows, columns, store)
    Threads.atomic_add!(assembler.calls, 1)
    return assembler.assembler(rows, columns, store)
end

@testset "BEAST block assembly" begin
    mesh = meshrectangle(1.0, 1.0, 0.4)
    space = raviartthomas(mesh)
    operator = Maxwell3D.singlelayer(; wavenumber=1.0)
    quadstrat = BEAST.DoubleNumSauterQstrat(2, 2, 0, 2, 2, 2)
    matrix = AdaptiveCrossApproximation.AbstractKernelMatrix(
        operator, space, space; matrixdata=quadstrat
    )
    calls = Threads.Atomic{Int}(0)
    counted = AdaptiveCrossApproximation.BEASTKernelMatrix{ComplexF64}(
        CountedBlockAssembler(matrix.nearassembler, calls)
    )

    # Unsorted indices and repeated rows preserve each requested block,
    # including requests with overlapping columns.
    rows = [[3, 1], [3, 1], [2, 4], [2, 4], [5]]
    columns = [[4, 1], [3, 2, 5], [1, 3], [3, 5], [2, 4]]
    references = [zeros(ComplexF64, length(r), length(c)) for (r, c) in zip(rows, columns)]
    for (block, r, c) in zip(references, rows, columns)
        matrix(block, r, c)
    end

    for scheduler in (SerialScheduler(), DynamicScheduler())
        # Assembly preserves accumulation and non-Matrix output layouts.
        blocks = AbstractMatrix{ComplexF64}[
            fill(1.0 + 2im, 2, 2),
            view(fill(1.0 + 2im, 4, 5), 2:3, 2:4),
            transpose(fill(1.0 + 2im, 2, 2)),
            fill(1.0 + 2im, 2, 2),
            fill(1.0 + 2im, 1, 2),
        ]
        calls[] = 0
        result = AdaptiveCrossApproximation.assemble_blocks(
            counted, blocks, rows, columns; scheduler
        )
        @test result === blocks
        @test calls[] == length(blocks)
        for (block, reference) in zip(blocks, references)
            @test block ≈ reference .+ (1.0 + 2im)
        end
    end

    # Ranges are valid index lists too; empty batches do no assembly work.
    ranges = [1:2, 1:2]
    rangecolumns = [1:2, 3:5]
    blocks = [zeros(ComplexF64, 2, length(c)) for c in rangecolumns]
    calls[] = 0
    AdaptiveCrossApproximation.assemble_blocks(counted, blocks, ranges, rangecolumns)
    @test calls[] == length(blocks)
    for (block, c) in zip(blocks, rangecolumns)
        reference = zeros(ComplexF64, 2, length(c))
        matrix(reference, 1:2, c)
        @test block ≈ reference
    end
    emptyblocks = Matrix{ComplexF64}[]
    calls[] = 0
    @test AdaptiveCrossApproximation.assemble_blocks(
        counted, emptyblocks, Vector{Int}[], Vector{Int}[]
    ) === emptyblocks
    @test calls[] == 0

    # General two-space assembly must not assume a symmetric matrix.
    trialspace = raviartthomas(translate(mesh, [0.2, 0.3, 1.0]))
    rectangular = AdaptiveCrossApproximation.AbstractKernelMatrix(
        operator, space, trialspace; matrixdata=BEAST.DoubleNumQStrat(2, 2)
    )
    blocks = [zeros(ComplexF64, length(r), length(c)) for (r, c) in zip(rows, columns)]
    AdaptiveCrossApproximation.assemble_blocks(
        rectangular, blocks, rows, columns; scheduler=DynamicScheduler()
    )
    for (block, r, c) in zip(blocks, rows, columns)
        reference = zeros(ComplexF64, length(r), length(c))
        rectangular(reference, r, c)
        @test block ≈ reference
    end
end
