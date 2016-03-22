#push!(LOAD_PATH, joinpath(dirname(@__FILE__), "../"))
#push!(LOAD_PATH, "C:/Users/hshindo/Documents/GitHub")
using Lapidary
#using Merlin

makedocs(
  modules = Merlin,
  clean = false
)

deploydocs(
  repo = "github.com/hshindo/Merlin.jl.git",
  julia  = "0.4"
)
