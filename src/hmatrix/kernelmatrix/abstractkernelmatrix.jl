abstract type AbstractKernelMatrix{T} end

function AbstractKernelMatrix(operator, testspace, trialspace; args...) end

function (::AbstractKernelMatrix)(matrixblock, tdata, sdata) end

Base.eltype(::AbstractKernelMatrix{T}) where {T} = T
