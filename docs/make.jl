using Documenter, DocumenterVitepress

using BigHOPs

cp(joinpath(@__DIR__, "benchmarks"), joinpath(@__DIR__, "src", "figures"); force=true)

makedocs(;
    modules=[BigHOPs],
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
