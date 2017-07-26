using HDF5
using Merlin

include("tagset.jl")
include("ner.jl")
include("model2.jl")

const wordembeds_file = ".data/glove.6B.100d.h5"
#const datapath = joinpath(dirname(@__FILE__), ".data")

# training
ner = NER()
train(ner, ".data/eng.train", ".data/eng.testb")
#save("NER.jld2", Dict("a"=>seg))

# decoding
#seg = load("NER.jld2")
#println(seg["a"].char2id)
#seg = Merlin.load("NER.merlin")
