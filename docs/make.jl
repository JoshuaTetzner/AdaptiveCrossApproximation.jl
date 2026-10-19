using Documenter
using AdaptiveCrossApproximation

DocMeta.setdocmeta!(
    AdaptiveCrossApproximation,
    :DocTestSetup,
    :(using AdaptiveCrossApproximation);
    recursive=true,
)

makedocs(;
    sitename="AdaptiveCrossApproximation.jl",
    authors="Joshua M. Tetzner <joshua.tetzner@uni-rostock.de> and contributors",
    modules=[AdaptiveCrossApproximation],
    # The API reference documents the internals as well.
    format=Documenter.HTML(; size_threshold=400 * 2^10, size_threshold_warn=300 * 2^10),
    pages=[
        "Introduction" => "index.md",
        "Manual" => Any[
            "Adaptive Cross Approximation" => "./manual/aca.md",
            "Incomplete Adaptive Cross Approximation" => "./manual/iaca.md",
            "Hierarchical Matrices" => "./manual/hmatrix.md",
        ],
        "Application Examples" => Any[
            "EFIE" => "./examples/efie.md",
            "MFIE" => "./examples/mfie.md",
            "PMCHWT" => "./examples/pmchwt.md",
            "Point Cloud" => "./examples/pointcloud.md",
        ],
        "Theory" => Any[
            "Adaptive Cross Approximation" => "./details/aca.md",
            "Incomplete Adaptive Cross Approximation" => "./details/iaca.md",
            "Hierarchical Matrices" => "./details/hmatrix.md",
        ],
        "Contributing" => "contributing.md",
        "API Reference" => "apiref.md",
    ],
)

deploydocs(;
    repo="github.com/JoshuaTetzner/AdaptiveCrossApproximation.jl.git",
    target="build",
    devbranch="dev",
    versions=["stable" => "v^", "dev" => "dev"],
)
