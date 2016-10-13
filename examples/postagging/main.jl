using Merlin
using HDF5

function main()
    h5file = "wordembeds_nyt100.h5"
    words = h5read(h5file, "s")
    wordembeds = Embedding(h5read(h5file,"v"))
    charembeds = Embedding(Float32,100,10)

    worddict = IntDict(words)
    chardict = IntDict{String}()
    tagdict = IntDict{String}()

    traindata = CoNLL.read(".data/wsj_00-18.conll", 2, 5)
    testdata = CoNLL.read(".data/wsj_22-24.conll", 2, 5)
    train_x, train_y = encode(traindata, worddict, chardict, tagdict, true)
    test_x, test_y = encode(testdata, worddict, chardict, tagdict, false)
    info("# words: $(length(worddict))")
    info("# chars: $(length(chardict))")
    info("# tags: $(length(tagdict))")

    model = Model(wordembeds, charembeds)
    # model = Merlin.load("postagger.h5", "model")
    train(5, model, train_x, train_y, test_x, test_y)

    Merlin.save("postagger.h5", "w", "model", model)
end

function train(nepochs::Int, model, train_x, train_y, test_x, test_y)
    opt = SGD()
    for epoch = 1:nepochs
        println("epoch: $(epoch)")
        opt.rate = 0.0075 / epoch
        loss = fit(train_x, train_y, model, crossentropy, opt)
        println("loss: $(loss)")

        test_z = map(x -> predict(model,x), test_x)
        acc = accuracy(test_y, test_z)
        println("test acc.: $(acc)")
        println("")
    end
end

predict(model, data) = argmax(model(data).data, 1)

function encode(data::Vector, worddict, chardict, tagdict, append::Bool)
    data_x, data_y = Vector{Token}[], Vector{Int}[]
    unkword = worddict["UNKNOWN"]
    for sent in data
        push!(data_x, Token[])
        push!(data_y, Int[])
        for items in sent
            word, tag = items[1], items[2]
            word0 = replace(word, r"[0-9]", '0')
            wordid = get(worddict, lowercase(word0), unkword)

            chars = Vector{Char}(word0)
            if append
                charids = map(c -> push!(chardict,string(c)), chars)
            else
                charids = map(c -> get(chardict,string(c),0), chars)
            end
            tagid = push!(tagdict, tag)
            token = Token(wordid, charids)
            push!(data_x[end], token)
            push!(data_y[end], tagid)
        end
    end
    data_x, data_y
end

function accuracy(golds::Vector{Vector{Int}}, preds::Vector{Vector{Int}})
    @assert length(golds) == length(preds)
    correct = 0
    total = 0
    for i = 1:length(golds)
        @assert length(golds[i]) == length(preds[i])
        for j = 1:length(golds[i])
            golds[i][j] == preds[i][j] && (correct += 1)
            total += 1
        end
    end
    correct / total
end

include("intdict.jl")
include("io.jl")
include("token.jl")
include("model.jl")

main()
