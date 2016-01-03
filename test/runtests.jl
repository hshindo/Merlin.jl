using Base.Test
using Merlin

tests = ["concat"]

println("Running tests...")

for t in tests
    path = joinpath(dirname(@__FILE__), "functors/$t.jl")
    println("$path ...")
    include(path)
end
