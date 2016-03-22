using Base.Test
using Merlin

println("Running tests...")

tests = ["concat"]
for t in tests
    path = joinpath(dirname(@__FILE__), "functors/$t.jl")
    println("$path ...")
    #include(path)
end
