using Documenter
using Merlin

makedocs(
    modules = [Merlin],
    clean = false,
    format = :html,
    sitename = "Merlin.jl",
    pages = [
        "Home" => "index.md",
        "Functions" => "functions.md",
        "Initializaters" => "initializers.md",
        #"Graph" => "graph.md",
        #"Optimizers" => "optimizers.md",
        #"Save and Load" => "save_load.md",
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
