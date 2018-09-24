using Test
using Merlin
using Merlin.CUDA

tests = ["functions"]

for t in tests
    path = joinpath(dirname(@__FILE__), "$t.jl")
    println("$path ...")
    include(path)
end
