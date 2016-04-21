using Documenter
using Merlin

makedocs(
  modules = [Merlin]
)

deploydocs(
  deps = deps   = Deps.pip("mkdocs", "python-markdown-math"),
  repo = "github.com/hshindo/Merlin.jl.git",
  julia  = "0.4"
)
