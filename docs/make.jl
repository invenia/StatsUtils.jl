using Documenter, StatsUtils

makedocs(
    modules=[StatsUtils],
    pages=[
        "Home" => "index.md",
        "API" => "api.md",
    ],
    repo="https://gitlab.invenia.ca/invenia/StatsUtils.jl/blob/{commit}{path}#L{line}",
    sitename="StatsUtils.jl",
    authors="Rory-Finnegan",
    assets=["assets/invenia.css"],
    strict = true,
    checkdocs = :none,
)
