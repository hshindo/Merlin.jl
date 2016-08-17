using Documenter
using Merlin

makedocs(
  modules = [Merlin]
)

deploydocs(
  #deps = Deps.pip("mkdocs", "mkdocs-material", "python-markdown-math", "pygments"),
  deps = Deps.pip("pygments", "mkdocs", "python-markdown-math", "mkdocs-material"),
  repo = "github.com/hshindo/Merlin.jl.git",
  julia  = "0.5",
)
