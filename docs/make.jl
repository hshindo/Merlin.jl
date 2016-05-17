using Documenter
using Merlin

makedocs(
  modules = [Merlin]
)

deploydocs(
  #deps = Deps.pip("mkdocs", "mkdocs-material", "python-markdown-math", "pygments"),
  deps = Deps.pip("mkdocs", "mkdocs-material", "pygments"),
  repo = "github.com/hshindo/Merlin.jl.git",
)
