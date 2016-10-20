ENV["USE_CUDA"] = true
using Base.Test
using Merlin
using JuCUDA

include("check.jl")

tests = ["functions"]

for t in tests
    path = joinpath(dirname(@__FILE__), "$t.jl")
    println("$path ...")
    include(path)
end
