struct PointMatrix{T,FunctorType,PointCollectionType} <: AbstractKernelMatrix{T}
    functor::FunctorType
    testpoints::PointCollectionType
    trialpoints::PointCollectionType
    function PointMatrix{T}(functor, testpoints, trialpoints) where {T}
        return new{T,typeof(functor),typeof(testpoints)}(functor, testpoints, trialpoints)
    end
end

function AdaptiveCrossApproximation.AbstractKernelMatrix(
    operator, testspace::AbstractVector, trialspace::AbstractVector; args...
)
    return AdaptiveCrossApproximation.PointMatrix{eltype(operator)}(
        operator, testspace, trialspace
    )
end

function AdaptiveCrossApproximation.AbstractKernelMatrix(
    operator::Function, testspace::AbstractVector, trialspace::AbstractVector; args...
)
    @warn "Using a plain function as kernel is not recommended."

    return AdaptiveCrossApproximation.PointMatrix{
        typeof(operator(testspace[1], trialspace[1]))
    }(
        operator, testspace, trialspace
    )
end

function (blk::PointMatrix)(matrixblock, tdata, sdata)
    for (i, t) in enumerate(tdata)
        for (j, s) in enumerate(sdata)
            matrixblock[i, j] += blk.functor(blk.testpoints[t], blk.trialpoints[s])
        end
    end
end

function Base.size(M::PointMatrix, dim=nothing)
    if dim === nothing
        return (length(M.testpoints), length(M.trialpoints))
    elseif dim == 1
        return length(M.testpoints)
    elseif dim == 2
        return length(M.trialpoints)
    else
        error("dim must be either 1 or 2")
    end
end

function nextrc!(buf, A::PointMatrix, i, j)
    for ii in eachindex(i)
        for jj in eachindex(j)
            buf[ii, jj] += A.functor(A.testpoints[i[ii]], A.trialpoints[j[jj]])
        end
    end
end
