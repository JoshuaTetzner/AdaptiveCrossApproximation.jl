"""
    GPUMatrixData{QuadStratType}

Matrix-data marker requesting GPU-accelerated near-field assembly.

Wraps a `quadstrat` together with `ndevices`, the number of GPU devices to
distribute near-field work across; recognized by the `ACABEASTCUDA` extension
when building a [`GPUBEASTKernelMatrix`](@ref).
"""
struct GPUMatrixData{QuadStratType}
    quadstrat::QuadStratType
    ndevices::Int
    function GPUMatrixData(quadstrat, ndevices::Int)
        return new{typeof(quadstrat)}(quadstrat, ndevices)
    end
end
