using Documenter, StatsUtils

makedocs(
    modules=[StatsUtils],
    format=:html,
    pages=[
        "Home" => "index.md",
    ],
    repo="https://gitlab.invenia.ca/invenia/StatsUtils.jl/blob/{commit}{path}#L{line}",
    sitename="StatsUtils.jl",
    authors="Rory-Finnegan",
    assets=["assets/invenia.css"],
)
