using BEAST

struct AbstractKernel{K}
    blockassembler::Function
end

function AbstractKernel(
    operator::BEAST.IntegralOperator,
    testspace::BEAST.Space,
    trialspace::BEAST.Space;
    quadstrat=BEAST.defaultquadstrat(operator, testspace, trialspace),
)
    return AbstractKernel{scalartype(operator)}(
        BEAST.blockassembler(operator, testspace, trialspace; quadstrat=quadstrat)
    )
end

function (M::AbstractKernel{K})(
    buf::AbstractArray{K}, i::AbstractArray{Int,1}, j::AbstractArray{Int,1}
) where {K}
    @views store(v, m, n) = (buf[m, n] += v)
    return M.blockassembler(i, j, store)
end

AdaptiveCrossApproximation.nextrc!(buf, A::AbstractKernel, i, j) = A(buf, i, j)
