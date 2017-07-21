using HDF5
using Merlin

type Segmenter
    word2id::Dict
    char2id::Dict
    tag2id::Dict
    id2tag::Dict
    nn
end

function Segmenter(ntags::Int)
    wordembeds_file = ".data/glove.6B.100d.h5"
    #wordembeds_file = ".data/word2vec_nyt100d.h5"
    words = h5read(wordembeds_file, "key")
    word2id = Dict(words[i] => i for i=1:length(words))
    char2id = Dict{String,Int}()
    tag2id = Dict{String,Int}()
    id2tag = Dict{Int,String}()
    wordembeds = h5read(wordembeds_file, "value")
    nn = Model(wordembeds, ntags)
    Segmenter(word2id, char2id, tag2id, id2tag, nn)
end

function decode(seg::Segmenter, words::Vector{String})
    wordids = map(w -> get(seg.word2id,w,0), words)
    charids = map(words) do w
        chars = Vector{Char}(w)
        
    end
    w, c = data
    y = seg.nn(w, c)
    vec(argmax(y.data,1))
end

include("data.jl")
include("model.jl")
include("eval.jl")

# training
using JLD2
#seg = Segmenter(6)
#datapath = joinpath(dirname(@__FILE__), ".data")
#train(seg, "$(datapath)/eng.train", "$(datapath)/eng.testb")
#save("NER.jld2", Dict("a"=>seg))

# decoding
seg = load("NER.jld2")
println(seg["a"].char2id)
#seg = Merlin.load("NER.merlin")
