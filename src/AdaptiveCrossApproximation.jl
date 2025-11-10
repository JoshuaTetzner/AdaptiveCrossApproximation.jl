module AdaptiveCrossApproximation

using LinearAlgebra
using StaticArrays

include("pivoting/abstractpivoting.jl")
include("pivoting/maxvalue.jl")
include("pivoting/lejapoints.jl")
include("pivoting/filldistance.jl")
include("pivoting/mimicrypivoting.jl")
include("pivoting/treemimicrypivoting.jl")

include("convergence/abstractconvergence.jl")
include("convergence/estimation.jl")
include("convergence/extrapolation.jl")
include("convergence/randomsampling.jl")
include("convergence/combinedconvcrit.jl")

include("pivoting/combinedpivstrat.jl")
include("pivoting/randomsampling.jl")

include("aca.jl")
include("acaT.jl")
include("iaca.jl")

if !isdefined(Base, :get_extension) # for julia version < 1.9
    include("../ext/ACAH2Trees/ACAH2Trees.jl")
end

export ACA
export iACA
export FNormEstimator, iFNormEstimator
export FNormExtrapolator
export MaximumValue
export Leja2
export FillDistance
export MimicryPivoting, TreeMimicryPivoting
end
