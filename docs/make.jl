using Documenter, DocumenterVitepress, DocumenterCitations

using BigHOPs

cp(joinpath(@__DIR__, "benchmarks"), joinpath(@__DIR__, "src", "figures"); force=true)

bib = CitationBibliography(joinpath(@__DIR__, "src", "references.bib"); style=:authoryear)

makedocs(;
    modules=[BigHOPs],
    plugins=[bib],
    authors="Domas Linkevicius",
    repo="https://github.com/dom-linkevicius/BigHOPs.jl",
    sitename="BigHOPs.jl",
    format=DocumenterVitepress.MarkdownVitepress(
        repo="https://github.com/dom-linkevicius/BigHOPs.jl",
        devurl="dev",
        deploy_url="https://dom-linkevicius.github.io/BigHOPs.jl",
    ),
    pages=[
        "Home" => "index.md",
        "Introduction" => "introduction.md",
        "Sample Code" => "sample_code.md",
        "Executors" => [
            "Executors" => "executors.md",
            "General functionalities" => "general_functionalities.md",
        ],
        "Samplers" => [
            "Samplers" => "samplers.md",
            "Benchmarks" => "benchmarks.md",
        ],
        "Plotting" => "plotting.md",
        "Contributing" => "contributing.md",
        "Provenance" => "provenance.md",
        "References" => "references.md",
        "API" => "api.md",
    ],
    warnonly=true,
)

DocumenterVitepress.deploydocs(;
    repo="github.com/dom-linkevicius/BigHOPs.jl",
    push_preview=true,
)
