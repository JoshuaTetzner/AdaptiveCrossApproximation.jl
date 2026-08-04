"""
    GPUMatrixData{QuadStratType}

Configuration for GPU-accelerated BEAST kernel matrix assembly.

Passed as the `matrixdata` keyword argument to [`AbstractKernelMatrix`](@ref) to build
a [`GPUBEASTKernelMatrix`](@ref) instead of a plain `BEASTKernelMatrix`; requires the
ACABEASTCUDA package extension to be loaded (BEAST.jl and CUDA.jl available).

Physical device selection is automatic: unless `devices` is given explicitly, the
extension picks `ndevices` currently-idle GPUs (no active compute processes) at
[`AdaptiveCrossApproximation.beastkernelmatrix`](@ref) build time. Devices are
dispatched via `OhMyThreads.@tasks` (`ntasks == ndevices`), so `ndevices` is not
tied to `Threads.nthreads()` - CPU-side code is free to use more threads than
there are devices; with fewer Julia threads than `ndevices`, devices simply lose
some overlap rather than erroring.

# Fields

  - `quadstrat::QuadStratType`: BEAST quadrature strategy used to assemble matrix entries
  - `ndevices::Int`: Number of GPU devices to use. When `devices` is given, defaults to
    `length(devices)`; otherwise defaults to `1`. Passing it explicitly alongside
    `devices` is allowed only if it matches `length(devices)`.
  - `devices::Union{Nothing,Vector{Int}}`: Explicit CUDA device ids to pin to, bypassing
    automatic free-device selection; `nothing` (default) means auto-select
"""
struct GPUMatrixData{QuadStratType}
    quadstrat::QuadStratType
    ndevices::Int
    devices::Union{Nothing,Vector{Int}}
    function GPUMatrixData(quadstrat; ndevices::Union{Int,Nothing}=nothing, devices=nothing)
        if devices !== nothing
            if ndevices !== nothing && ndevices != length(devices)
                throw(ArgumentError(
                    "ndevices=$ndevices does not match length(devices)=$(length(devices))"
                ))
            end
            ndevices = length(devices)
        elseif ndevices === nothing
            ndevices = 1
        end
        ndevices > 0 || throw(ArgumentError("ndevices must be positive"))
        return new{typeof(quadstrat)}(quadstrat, ndevices, devices)
    end
end

"""
    GPUBEASTKernelMatrix{T,NearBlockAssemblerType,FunctorsType} <: AbstractKernelMatrix{T}

GPU-accelerated kernel matrix wrapper for BEAST operator assembly.

Analogous to [`BEASTKernelMatrix`](@ref), but backed by a GPU near-field block
assembler. Built automatically by `AdaptiveCrossApproximation.beastkernelmatrix` when
[`GPUMatrixData`](@ref) is passed as `matrixdata`; requires the ACABEASTCUDA package
extension to be loaded.

# Fields

  - `nearassembler::NearBlockAssemblerType`: `functors[1]`, used for single-block
    sampling (the per-block call operator / classic ACA pivoting via `nextrc!`), which
    is inherently single-device
  - `functors::FunctorsType`: one GPU-backed BEAST assembler per device (length 1 for
    `ndevices == 1`); used by the batched `assemble_blocks` path, which parallelizes
    across all of them

# Type parameters

  - `T`: scalar element type returned by kernel evaluations
  - `NearBlockAssemblerType`: type of `nearassembler` (a single device's assembler)
  - `FunctorsType`: type of `functors` (a vector of per-device assemblers)
"""
struct GPUBEASTKernelMatrix{T,NearBlockAssemblerType,FunctorsType} <: AbstractKernelMatrix{T}
    nearassembler::NearBlockAssemblerType
    functors::FunctorsType
    function GPUBEASTKernelMatrix{T}(functors) where {T}
        return new{T,eltype(functors),typeof(functors)}(functors[1], functors)
    end
end

function Base.size(matrix::GPUBEASTKernelMatrix, dim=nothing)
    if dim === nothing
        return (length(matrix.nearassembler.test_space), length(matrix.nearassembler.trial_space))
    elseif dim == 1
        return length(matrix.nearassembler.test_space)
    elseif dim == 2
        return length(matrix.nearassembler.trial_space)
    else
        error("dim must be either 1 or 2")
    end
end

function nextrc!(buffer, matrix::GPUBEASTKernelMatrix, rows, columns)
    return matrix(buffer, rows, columns)
end
