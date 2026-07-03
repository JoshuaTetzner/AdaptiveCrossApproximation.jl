"""
    GPUMatrixData{QuadStratType}

Configuration for GPU-accelerated BEAST kernel matrix assembly.

Passed as the `matrixdata` keyword argument to [`AbstractKernelMatrix`](@ref) to build
a [`GPUBEASTKernelMatrix`](@ref) instead of a plain `BEASTKernelMatrix`; requires the
ACABEASTCUDA package extension to be loaded (BEAST.jl and CUDA.jl available).

# Fields

  - `quadstrat::QuadStratType`: BEAST quadrature strategy used to assemble matrix entries
  - `ndevices::Int`: Number of GPU devices to distribute the assembly across
"""
struct GPUMatrixData{QuadStratType}
    quadstrat::QuadStratType
    ndevices::Int
    function GPUMatrixData(quadstrat, ndevices::Int)
        return new{typeof(quadstrat)}(quadstrat, ndevices)
    end
end
