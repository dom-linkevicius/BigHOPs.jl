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
        "Getting Started" => "getting_started.md",
        "Executors" => "executors.md",
        "Samplers" => "samplers.md",
        "Plotting" => "plotting.md",
        "Benchmarks" => "benchmarks.md",
        "Provenance" => "provenance.md",
        "References" => "references.md",
        "Contributing" => "contributing.md",
        "API" => "api.md",
    ],
    warnonly=true,
)

DocumenterVitepress.deploydocs(;
    repo="github.com/dom-linkevicius/BigHOPs.jl",
    push_preview=true,
)
