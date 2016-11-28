using Merlin
using MLDatasets
using HDF5

function main()
    h5file = "wordembeds_nyt100.h5"
    words = h5read(h5file, "s")
    wordembeds = Lookup(h5read(h5file,"v"))
    charembeds = Lookup(Float32,100,10)

    worddict = IntDict(words)
    chardict = IntDict{String}()
    tagdict = IntDict{String}()

    #traindata = UD_English.traindata()
    #testdata = UD_English.testdata()
    #traindata = CoNLL.read(".data/wsj_00-18.conll")
    #testdata = CoNLL.read(".data/wsj_22-24.conll")
    traindata = CoNLL.read(".data/eng.train.IOBES")
    testdata = CoNLL.read(".data/eng.testb.IOBES")
    info("# sentences of train data: $(length(traindata))")
    info("# sentences of test data: $(length(testdata))")

    train_x, train_y = encode(traindata, worddict, chardict, tagdict, true)
    test_x, test_y = encode(testdata, worddict, chardict, tagdict, false)
    info("# words: $(length(worddict))")
    info("# chars: $(length(chardict))")
    info("# tags: $(length(tagdict))")

    model = Model(wordembeds, charembeds, length(tagdict))
    # model = Merlin.load("postagger.h5", "model")
    train(10, model, train_x, train_y, test_x, test_y, tagdict)

    #Merlin.save("postagger.h5", "w", "model", model)
end

function train(nepochs::Int, model, train_x, train_y, test_x, test_y, tagdict)
    opt = SGD(0.01)
    for epoch = 1:nepochs
        println("epoch: $(epoch)")
        #opt.rate = 0.01 / epoch
        loss = fit(train_x, train_y, model, crossentropy, opt)
        println("loss: $(loss)")

        test_z = map(x -> predict(model,x), test_x)
        acc = accuracy(test_y, test_z)
        println("test acc.: $(acc)")

        prec, recall, fval = fvalue(test_y, test_z, tagdict)
        println("prec: $(prec)")
        println("recall: $(recall)")
        println("fvalue: $(fval)")
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
            @assert length(items) == 2
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

function fvalue(golds::Vector{Vector{Int}}, preds::Vector{Vector{Int}}, tagdict)
    bio = BIO()
    tagset = map(collect(1:length(tagdict))) do i
        str = getkey(tagdict,i)
        items = split(str, '-') # B-ORG -> B
        decode(bio, String(items[1]))
    end

    @assert length(golds) == length(preds)
    correct = 0
    total_g, total_p = 0, 0
    for i = 1:length(golds)
        @assert length(golds[i]) == length(preds[i])
        gold, pred = golds[i], preds[i]
        span_g = decode(bio, map(x -> tagset[x], gold))
        span_p = decode(bio, map(x -> tagset[x], pred))
        set = intersect(Set(span_g), Set(span_p))
        for span in set
            all(k -> gold[k] == pred[k], span) && (correct += 1)
        end
        total_g += length(span_g)
        total_p += length(span_p)
    end
    prec = correct / total_g
    recall = correct / total_p
    fval = 2*recall*prec/(recall+prec)
    prec, recall, fval
end

function fvalue(golds::Vector{Vector{UnitRange{Int}}}, preds::Vector{Vector{UnitRange{Int}}})
    @assert length(golds) == length(preds)
    correct = 0
    total_y, total_z = 0, 0
    for i = 1:length(golds)
        s1 = Set(golds[i])
        s2 = Set(preds[i])
        correct += length(intersect(s1,s2))
        total_y += length(s1)
        total_z += length(s2)
    end
    prec = correct / total_y
    recall = correct / total_z
    fval = 2*recall*prec/(recall+prec)
    prec, recall, fval
end

include("intdict.jl")
include("tagset.jl")
include("token.jl")
include("model.jl")

main()
