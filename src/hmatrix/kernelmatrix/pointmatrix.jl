"""
    PointMatrix{T,FunctorType,PointCollectionType} <: AbstractKernelMatrix{T}

Kernel matrix evaluating entries from a kernel functor and point collections.

Stores point sample locations and a kernel function, computing matrix entries
on demand by evaluating the kernel at test and trial point pairs. Useful for
point cloud interactions or problems with explicit coordinate data.

# Fields

  - `functor::FunctorType`: Kernel function or operator to evaluate
  - `testpoints::PointCollectionType`: Test space point locations
  - `trialpoints::PointCollectionType`: Trial space point locations

# Type parameters

  - `T`: scalar element type returned by kernel evaluations
  - `FunctorType`: type of the kernel functor
  - `PointCollectionType`: type of point collections (e.g., Vector, AbstractArray)
"""
struct PointMatrix{T,FunctorType,PointCollectionType} <: AbstractKernelMatrix{T}
    functor::FunctorType
    testpoints::PointCollectionType
    trialpoints::PointCollectionType
    function PointMatrix{T}(functor, testpoints, trialpoints) where {T}
        return new{T,typeof(functor),typeof(testpoints)}(functor, testpoints, trialpoints)
    end
end

"""
    AbstractKernelMatrix(operator, testspace::AbstractVector, trialspace::AbstractVector; args...)

Create a `PointMatrix` kernel wrapper from vector point data and operator.

Constructs a `PointMatrix` when given vector collections of test and trial points.
Matches the exported `AbstractKernelMatrix` factory interface to provide point-based
kernel evaluation.

# Arguments

  - `operator`: kernel function or operator
  - `testspace::AbstractVector`: test point locations (indexed collection)
  - `trialspace::AbstractVector`: trial point locations (indexed collection)
  - `args...`: additional keyword arguments (unused for point-based evaluation)

# Returns

  - `PointMatrix` with element type inferred from `operator`
"""
function AdaptiveCrossApproximation.AbstractKernelMatrix(
    operator, testspace::AbstractVector, trialspace::AbstractVector; args...
)
    return AdaptiveCrossApproximation.PointMatrix{eltype(operator)}(
        operator, testspace, trialspace
    )
end

"""
    AbstractKernelMatrix(operator::Function, testspace::AbstractVector, trialspace::AbstractVector; args...)

Create a `PointMatrix` from a plain Julia function kernel (with warning).

Provides fallback support for plain function kernels. Issues a warning as
operators with structure (kernels, boundary integral operators) are typically
preferred for better performance and correctness.

# Arguments

  - `operator::Function`: plain Julia function with signature `operator(testpoint, trialpoint)`
  - `testspace::AbstractVector`: test point locations (indexed collection)
  - `trialspace::AbstractVector`: trial point locations (indexed collection)
  - `args...`: additional keyword arguments (unused)

# Returns

  - `PointMatrix` with element type inferred from function evaluation
"""
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
