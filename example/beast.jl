using BEAST
using CompScienceMeshes
using AdaptiveCrossApproximation
using StaticArrays

#Define helper Functions

m1 = meshrectangle(1.0, 1.0, 0.1)
m2 = translate(meshrectangle(1.0, 1.0, 0.1), SVector(2.0, 0.0, 0.0))

op = Helmholtz3D.singlelayer()
sp1 = lagrangecxd0(m1)
sp2 = lagrangecxd0(m2)

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
    buf::AbstractArray{K}, i::AbstractArray{Int,1}, j::AbstractArray{Int,1}
) where {K}
    @views store(v, m, n) = (buf[m, n] += v)
    return M.blockassembler(i, j, store)
end

K = AbstractKernel(op, sp1, sp2);

AdaptiveCrossApproximation.nextrc!(buf, A::AbstractKernel, i, j) = A(buf, i, j)

##
aca = ACA(
    AdaptiveCrossApproximation.MaximumValue(zeros(Bool, length(sp1.pos))),
    AdaptiveCrossApproximation.MaximumValue(zeros(Bool, length(sp2.pos))),
    AdaptiveCrossApproximation.FNormEstimator(0.0),
)

rowbuffer = zeros(Float64, 50, length(sp2.pos))
colbuffer = zeros(Float64, length(sp1.pos), 50)

@time npivots = aca(K, rowbuffer, colbuffer, 50, 1e-4);
