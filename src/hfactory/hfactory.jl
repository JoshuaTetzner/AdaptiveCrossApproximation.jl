"""
    LowRankCompression(compressor=ACA(); maxrank=40)

Far-field strategy for `HFactory.HMatrixConstruction`: compress each far-field block
with the ACA-style `compressor` into a [`LowRankMatrix`](@ref) of rank at most
`maxrank`. Sources can be matrices or [`AbstractKernelMatrix`](@ref) objects. With the
near strategy `nothing`, near-field blocks of kernel matrices are assembled through
their block interface.

```julia
scheme = HFactory.HMatrixConstruction(nothing, LowRankCompression(ACA(; tol=1e-6)))
H = HFactory.assemble(kernel, index, scheme)
```
"""
struct LowRankCompression{C}
    compressor::C
    maxrank::Int
end

LowRankCompression(compressor=ACA(); maxrank::Int=40) =
    LowRankCompression(compressor, maxrank)

include("construction.jl")
include("matvec.jl")
