push!(LOAD_PATH, joinpath(dirname(@__FILE__), "../../.."))
push!(LOAD_PATH, dirname(@__FILE__))
ENV["USE_CUDA"] = true
workspace()
using Merlin

function bench()
  a =  Array{Float32,2}[]
  push!(a, zeros(Float32,100,100))
  push!(a, zeros(Float32,100,100))
  for i = 1:1000
    Merlin.concat(a, 1)
    #cat(1,a...)
  end
end

@time bench()

include("token.jl")
include("model.jl")
include("train.jl")
path = "C:/Users/shindo/Dropbox/tagging"

@time train(path)
