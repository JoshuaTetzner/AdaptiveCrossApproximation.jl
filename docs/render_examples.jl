# Regenerate the pre-rendered plot assets embedded in the documentation.
# Run from the package root:
#
#   julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate(); include("docs/render_examples.jl")'
#
# Commit the generated files in docs/src/assets/examples/ afterwards.

using LinearAlgebra
using CompScienceMeshes
using BEAST
using ParallelKMeans
using H2Trees
using AdaptiveCrossApproximation
using Krylov
using PlotlyBase

outdir = joinpath(@__DIR__, "src", "assets", "examples")
mkpath(outdir)

include(joinpath(@__DIR__, "..", "example", "efie.jl"))
open(joinpath(outdir, "efie_results.html"), "w") do io
    return PlotlyBase.to_html(io, plt)
end
include(joinpath(@__DIR__, "..", "example", "mfie.jl"))
open(joinpath(outdir, "mfie_results.html"), "w") do io
    return PlotlyBase.to_html(io, plt)
end
include(joinpath(@__DIR__, "..", "example", "pmchwt.jl"))
open(joinpath(outdir, "pmchwt_results.html"), "w") do io
    return PlotlyBase.to_html(io, plt)
end

@info "Plots written to $outdir"
