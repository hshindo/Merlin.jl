workspace()
using Merlin

include("token.jl")
include("model_char.jl")
include("train.jl")
path = "C:/Users/hshindo/Dropbox/tagging"

@time train(path)
