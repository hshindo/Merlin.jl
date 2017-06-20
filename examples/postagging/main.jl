using Merlin
using MLDatasets
using HDF5

const h5file = "wordembeds_nyt100.h5"

function setup_data(doc::Vector, worddict::Dict, chardict::Dict, tagdict::Dict)
    data_w, data_c, data_t = Vector{Int}[], Vector{Vector{Int}}[], Vector{Int}[]
    unkword = worddict["UNKNOWN"]
    for sent in doc
        w = Int[]
        c = Vector{Int}[]
        t = Int[]
        for items in sent
            word, tag = items[2], items[5]
            word0 = replace(word, r"[0-9]", '0')
            wordid = get(worddict, lowercase(word0), unkword)
            chars = Vector{Char}(word0)
            charid = map(c -> get!(chardict,c,length(chardict)+1), chars)
            tagid = get!(tagdict, tag, length(tagdict)+1)
            push!(w, wordid)
            push!(c, charid)
            push!(t, tagid)
        end
        push!(data_w, w)
        push!(data_c, c)
        push!(data_t, t)
    end
    data_w, data_c, data_t
end

function train()
    words = h5read(h5file, "s")
    worddict = Dict(words[i] => i for i=1:length(words))
    chardict = Dict{Char,Int}()
    tagdict = Dict{String,Int}()
    #traindoc = UD_English.traindata()
    #testdoc = UD_English.testdata()
    traindoc = CoNLL.read(".data/wsj_00-18.conll")
    testdoc = CoNLL.read(".data/wsj_22-24.conll")
    info("# sentences of train doc: $(length(traindoc))")
    info("# sentences of test doc: $(length(testdoc))")

    train_w, train_c, train_t = setup_data(traindoc, worddict, chardict, tagdict)
    test_w, test_c, test_t = setup_data(testdoc, worddict, chardict, tagdict)
    info("# words: $(length(worddict))")
    info("# chars: $(length(chardict))")
    info("# tags: $(length(tagdict))")

    nn = create_model(length(tagdict))
    opt = SGD()
    for epoch = 1:10
        println("epoch: $epoch")
        opt.rate = 0.0075 / epoch
        loss = minimize!(train_f, (train_w,train_c), train_t, opt)
        println("loss: $loss")

        ys = cat(1, map(x -> vec(x.data), test_y)...)
        zs = cat(1, map(x -> vec(model(x).data), test_x)...)
        acc = mean(i -> ys[i] == zs[i] ? 1.0 : 0.0, 1:length(ys))
        println("test acc.: $acc")
        println()
    end
end

include("model.jl")
train()
