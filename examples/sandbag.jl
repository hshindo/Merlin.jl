push!(LOAD_PATH, joinpath(dirname(@__FILE__), "../.."))
push!(LOAD_PATH, dirname(@__FILE__))
ENV["USE_CUDA"] = true
workspace()
using Merlin
using JLD

type AAA
  a
end

path = "C:/temp/"
A = reshape(1:120, 15, 8)
A = AAA(A)
save("$(path)/A.jld", "A", A)
v = load("$(path)/A.jld", "A")
