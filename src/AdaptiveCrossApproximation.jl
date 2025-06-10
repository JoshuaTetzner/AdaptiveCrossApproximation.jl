module AdaptiveCrossApproximation

using LinearAlgebra
using StaticArrays

include("pivoting/abstractpivoting.jl")
include("pivoting/maxvalue.jl")

include("convergence/abstractconvergence.jl")
include("convergence/default.jl")

include("aca.jl")

export ACA

end
