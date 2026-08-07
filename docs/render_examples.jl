# Regenerate the pre-rendered plot assets embedded in the documentation.
# Run from the package root:
#
#   julia --project=docs docs/render_examples.jl
#
# Commit the generated files in docs/src/assets/examples/ afterwards.

using LinearAlgebra
using CompScienceMeshes
using BEAST
using ParallelKMeans
using H2Trees
using AdaptiveCrossApproximation
using Krylov
using PlotlyJS

outdir = joinpath(@__DIR__, "src", "assets", "examples")
mkpath(outdir)
ENV["ACA_OUTPUT_DIR"] = outdir

include(joinpath(@__DIR__, "..", "example", "efie.jl"))
include(joinpath(@__DIR__, "..", "example", "mfie.jl"))
include(joinpath(@__DIR__, "..", "example", "pmchwt.jl"))

@info "Plots written to $outdir"
