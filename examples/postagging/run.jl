github_path = "C:/Users/shindo/Documents/GitHub"
push!(LOAD_PATH, github_path)
push!(LOAD_PATH, "$(github_path)/Merlin.jl/examples/postagging")
workspace()
using Merlin
using POSTagging
path = "C:/Users/shindo/Dropbox/tagging"

g = POSTagging.posmodel(path)
dicts = (Dict(), Dict(), Dict())
tokens = POSTagging.readCoNLL("$(path)/wsj_00-18.conll", dicts)[1]
words = map(t -> t.word, tokens)
chars = convert(Vector{Vector{Char}}, map(w -> convert(Vector{Char}, w), words))
chars = map(cs -> [' ', ' ', cs..., ' ', ' '], chars)
TensorNet.apply(g, (words, chars))

function bench()
  for i = 1:10000
    TensorNet.apply(g, chars)
    #for j = 1:length(tokens)
    #  TensorNet.apply(g, chars2[j])
    #end
  end
end

@time bench()

@time POSTagging.train(path)
