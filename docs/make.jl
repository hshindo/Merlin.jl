using Documenter
using Merlin

makedocs(
    modules = [Merlin],
    format = :html,
    sitename = "Merlin.jl",
    pages = [
        "Home" => "index.md",
        "Var" => "var.md",
        "Functions" => "functions.md",
        "Graph" => "graph.md",
        "Initializaters" => "initializers.md",
        "Optimizers" => "optimizers.md",
        "Datasets" => "datasets.md",
        "Save and Load" => "save_load.md",
    ]
)

deploydocs(
    # deps = Deps.pip("pygments", "mkdocs", "python-markdown-math"),
    # deps = Deps.pip("pygments", "mkdocs", "python-markdown-math", "mkdocs-material"),
    repo = "github.com/hshindo/Merlin.jl.git",
    julia  = "0.6",
    target = "build",
    deps = nothing,
    make = nothing,
)
