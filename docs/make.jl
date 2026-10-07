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
    pages=[
        "Introduction" => "index.md",
        "Manual" => Any[
            "Adaptive Cross Approximation" => "./manual/aca.md",
            "Incomplete Adaptive Cross Approximation" => "./manual/iaca.md",
            "Hierarchical Matrices" => "./manual/hmatrix.md",
            "Application Examples" => "./manual/examples.md",
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
