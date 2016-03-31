push!(LOAD_PATH, joinpath(dirname(@__FILE__), "../../.."))
push!(LOAD_PATH, dirname(@__FILE__))
ENV["USE_CUDA"] = true
workspace()
using Merlin

include("token.jl")
include("model_char.jl")
include("train.jl")
path = "C:/Users/hshindo/Dropbox/tagging"

@time train(path)
