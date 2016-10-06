using Documenter
using Merlin

makedocs(
    modules = [Merlin],
    format   = Documenter.Formats.HTML,
    sitename = "Merlin.jl",
    pages = Any[
        "Home" => "index.md",
        "Functions" => "functions.md",
        #"Graph" => "graph.md",
        "Optimizers" => "optimizers.md",
        "Save and Load" => "save_load.md",
    ]
)

deploydocs(
    deps = Deps.pip("pygments", "mkdocs", "python-markdown-math"),
    #deps = Deps.pip("pygments", "mkdocs", "python-markdown-math", "mkdocs-material"),
    repo = "github.com/hshindo/Merlin.jl.git",
    julia  = "0.5",
    target = "build",
    deps = nothing,
    make = nothing,
)
