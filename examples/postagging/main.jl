using Merlin
using MLDatasets
using HDF5

const h5file = "wordembeds_nyt100.h5"

function setup_data()
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

    train_x, train_y = setup_data(traindoc, worddict, chardict, tagdict)
    test_x, test_y = setup_data(testdoc, worddict, chardict, tagdict)
    info("# words: $(length(worddict))")
    info("# chars: $(length(chardict))")
    info("# tags: $(length(tagdict))")
    train_x, train_y, test_x, test_y, worddict, chardict, tagdict
end

function setup_data(doc::Vector, worddict, chardict, tagdict)
    data_x, data_y = Tuple{Var,Vector{Var}}[], Var[]
    unkword = worddict["UNKNOWN"]
    for sent in doc
        w = Int[]
        cs = Var[]
        t = Int[]
        for items in sent
            word, tag = items[2], items[5]
            word0 = replace(word, r"[0-9]", '0')
            wordid = get(worddict, lowercase(word0), unkword)
            chars = Vector{Char}(word0)
            charids = map(c -> get!(chardict,c,length(chardict)+1), chars)
            tagid = get!(tagdict, tag, length(tagdict)+1)
            push!(w, wordid)
            push!(cs, Var(reshape(charids,1,length(charids))))
            push!(t, tagid)
        end
        w = reshape(w, 1, length(w))
        t = reshape(t, 1, length(t))
        push!(data_x, (Var(w),cs))
        push!(data_y, Var(t))
    end
    data_x, data_y
end

function train()
    train_x, train_y, test_x, test_y, worddict, chardict, tagdict = setup_data()
    wordembeds = h5read(h5file, "v")
    charembeds = rand(Float32, 10, 100)
    model = Model(wordembeds, charembeds, length(tagdict))

    opt = SGD()
    for epoch = 1:10
        println("epoch: $epoch")
        opt.rate = 0.0075 / epoch
        loss = fit(train_x, train_y, model, opt)
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
