# Contributing

In order to contribute to this package directly create a pull request against the `dev` branch (the integration branch; `main` is the stable/release branch, updated only by merging `dev` into it). Before doing so please: 

- Follow the style of the surrounding code.
- Supplement the documentation.
- Write tests and check that no errors occur.


---
## Style

For a consistent style the [JuliaFormatter.jl](https://github.com/domluna/JuliaFormatter.jl) package is used which enforces the style defined in the *.JuliaFormatter.toml* file. To follow this style simply run
```julia
using JuliaFormatter
format(pkgdir(AdaptiveCrossApproximation))
```

!!! note
    That all files follow the JuliaFormatter style is tested during the unit tests. Hence, do not forget to execute the two lines above. Otherwise, the tests are likely to not pass.


---
## Documentation

Add documentation for any changes or new features following the style of the existing documentation. For more information you can have a look at the [Documenter.jl](https://documenter.juliadocs.org/stable/) documentation.


---
## Documentation plots

The interactive plots on the [Application Examples](@ref) page are pre-rendered HTML
files stored in `docs/src/assets/examples/` and committed to the repository; the docs
build only serves these static files, it does not re-run the examples. After changing
`example/efie.jl` or `example/mfie.jl`, regenerate them from the package root:

```sh
julia --project=docs docs/render_examples.jl
```

and commit the updated HTML files alongside your example changes.


---
## [Tests](@id tests)

Write tests for your code changes and verify that no errors occur, e.g., by running
```julia
using Pkg
Pkg.test("AdaptiveCrossApproximation.jl")
```
For a detailed information on which parts are tested the coverage can be evaluated on your local machine, e.g., by
```julia
using Pkg
Pkg.test("AdaptiveCrossApproximation"; coverage=true, julia_args=["-t 4"])

# determine coverage
using Coverage
src_folder = pkgdir(AdaptiveCrossApproximation) * "/src"
coverage   = process_folder(src_folder)
LCOV.writefile("path-to-folder-you-like" * "AdaptiveCrossApproximation.lcov.info", coverage)

clean_folder(src_folder) # delete .cov files

# extract information about coverage
covered_lines, total_lines = get_summary(coverage)
@info "Current coverage:\n$covered_lines of $total_lines lines ($(round(Int, covered_lines / total_lines * 100)) %)"
```

In Visual Studio Code the [Coverage Gutters](https://marketplace.visualstudio.com/items?itemName=ryanluker.vscode-coverage-gutters) plugin can be used to visualize the tested lines of the code by inserting the path of the *AdaptiveCrossApproximation.lcov.info* file in the settings.