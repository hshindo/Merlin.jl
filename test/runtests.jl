using Base.Test
using Merlin

a = rand(10, 5)
findmax(a, 1)

tests = ["concat"]

println("Running tests...")

for t in tests
    path = joinpath(dirname(@__FILE__), "functors/$t.jl")
    println("$path ...")
    include(path)
end
