using Base.Test
using Merlin
import Merlin.CuArray

include("check.jl")

tests = ["functions"]

for t in tests
    path = joinpath(dirname(@__FILE__), "$t.jl")
    println("$path ...")
    include(path)
end
