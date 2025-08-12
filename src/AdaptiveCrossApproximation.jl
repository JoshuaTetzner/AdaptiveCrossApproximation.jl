module AdaptiveCrossApproximation

using LinearAlgebra
using StaticArrays

include("pivoting/abstractpivoting.jl")
include("pivoting/maxvalue.jl")
include("pivoting/lejapoints.jl")
include("pivoting/filldistance.jl")

include("convergence/abstractconvergence.jl")
include("convergence/estimation.jl")
include("convergence/extrapolation.jl")
include("convergence/randomsampling.jl")
include("convergence/combinedconvcrit.jl")

include("pivoting/combinedpivstrat.jl")
include("pivoting/randomsampling.jl")

include("aca.jl")

if !isdefined(Base, :get_extension) # for julia version < 1.9
    include("../ext/ACAHMatrices/ACAHMatrices.jl")
end

export ACA
export FNormEstimator
export FNormExtrapolator
export MaximumValue
export Leja2
export FillDistance

end
