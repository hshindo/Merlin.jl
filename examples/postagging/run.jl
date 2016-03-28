push!(LOAD_PATH, joinpath(dirname(@__FILE__), "../../.."))
push!(LOAD_PATH, dirname(@__FILE__))
ENV["USE_CUDA"] = true
workspace()
using Merlin

function bench()
  for i = 1:100000
    #Variable()
    Array(Float32,100)
  end
end

@time begin
  @parallel for i = 1:1000000
    r = rand(Float32, 100, 100)
  end
end

@time bench()

include("token.jl")
include("model_char.jl")
include("train.jl")
path = "C:/Users/hshindo/Dropbox/tagging"

@time train(path)
