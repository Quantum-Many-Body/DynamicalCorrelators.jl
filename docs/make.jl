using Documenter
using DynamicalCorrelators

makedocs(;
    modules=[DynamicalCorrelators],
    authors="Y.-Y.Zong",
    repo="https://github.com/ZongYongyue/DynamicalCorrelators.jl/blob/{commit}{path}#{line}",
    sitename="DynamicalCorrelators.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ZongYongyue.github.io/DynamicalCorrelators.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Tutorials" => [
            "Getting Started" => "tutorials/getting_started.md",
            "Ground State with DMRG" => "tutorials/dmrg.md",
            "Dynamical Correlations" => "tutorials/dynamical_correlations.md",
            "Spectral Functions" => "tutorials/spectral_functions.md",
            "Finite Temperature" => "tutorials/finite_temperature.md",
        ],
        "API Reference" => [
            "Models & Lattices" => "api/models.md",
            "Operators" => "api/operators.md",
            "States" => "api/states.md",
            "Algorithms" => "api/algorithms.md",
            "Observables" => "api/observables.md",
            "Utilities" => "api/utilities.md",
        ],
    ],
    checkdocs=:none,
    warnonly=true,
)

deploydocs(;
    repo="github.com/ZongYongyue/DynamicalCorrelators.jl",
    devbranch="main",
    push_preview=true,
)
