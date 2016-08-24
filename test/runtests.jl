using Merlin
using Base.Test

import Merlin.CuArray

tests = ["functions", "graphs"]

for t in tests
    path = joinpath(dirname(@__FILE__), "$t.jl")
    println("$path ...")
    include(path)
end
