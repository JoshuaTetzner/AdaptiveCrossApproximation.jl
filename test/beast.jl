using BEAST

struct AbstractKernel{K}
    blockassembler::Function
end

function AbstractKernel(
    operator::BEAST.IntegralOperator, testspace::BEAST.Space, trialspace::BEAST.Space
)
    return AbstractKernel{scalartype(operator)}(
        BEAST.blockassembler(operator, testspace, trialspace)
    )
end

function (M::AbstractKernel{K})(
    buf::AbstractMatrix{K}, i::AbstractVector{Int}, j::AbstractVector{Int}
) where {K}
    @views store(v, m, n) = (buf[m, n] += v)
    return blkasm(i, j, store)
end
