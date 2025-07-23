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
            "General Usage" => "./manual/manual.md",
            "Application Examples" => "./manual/examples.md",
        ],
        "Further Details" =>
            Any["ACA" => "./details/aca.md", "iACA" => "./details/iaca.md"],
        "Contributing" => "contributing.md",
        "API Reference" => "apiref.md",
    ],
)

deploydocs(;
    repo="github.com/FastBEAST/AdaptiveCrossApproximation.jl.git",
    target="build",
    devbranch="dev",
    versions=["stable" => "v^", "dev" => "dev"],
)
