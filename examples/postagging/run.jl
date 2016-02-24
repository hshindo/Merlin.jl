push!(LOAD_PATH, joinpath(dirname(@__FILE__), "../../.."))
push!(LOAD_PATH, dirname(@__FILE__))
workspace()

using Merlin
using POSTagging
path = "C:/Users/hshindo/Dropbox/tagging"

@time POSTagging.train(path)

Profile.clear()
@profile POSTagging.train(path)
Profile.print()
