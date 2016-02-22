push!(LOAD_PATH, joinpath(dirname(@__FILE__), "../../.."))
push!(LOAD_PATH, dirname(@__FILE__))
ENV["USE_CUDA"] = true
workspace()
using IterativeSolvers
using ArrayFire
using Merlin
using CUDNN
using POSTagging
path = "C:/Users/hshindo/Dropbox/tagging"

set_backend("cpu")

@time POSTagging.train(path)

Profile.clear()
@profile POSTagging.train(path)
Profile.print()
