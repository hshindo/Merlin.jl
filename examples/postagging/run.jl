push!(LOAD_PATH, joinpath(dirname(@__FILE__), "../../.."))
push!(LOAD_PATH, dirname(@__FILE__))
workspace()

using Merlin
include("token.jl")
include("model.jl")
include("train.jl")
path = "C:/Users/hshindo/Dropbox/tagging"

@time train(path)
