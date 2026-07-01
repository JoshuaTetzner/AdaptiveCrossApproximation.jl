struct GPUMatrixData{QuadStratType}
    quadstrat::QuadStratType
    ndevices::Int
    function GPUMatrixData(quadstrat, ndevices::Int)
        return new{typeof(quadstrat)}(quadstrat, ndevices)
    end
end
