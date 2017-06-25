using Merlin
using HDF5
using ProgressMeter

const wordembeds_file = "wordembeds_nyt100.h5"

function train()
    words = h5read(wordembeds_file, "s")
    worddict = Dict(words[i] => i for i=1:length(words))
    chardict = Dict{Char,Int}()
    tagdict = Dict{String,Int}()
    #traindoc = UD_English.traindata()
    #testdoc = UD_English.testdata()
    train_w, train_c, train_t = readdata(".data/wsj_00-18.conll", worddict, chardict, tagdict)
    test_w, test_c, test_t = readdata(".data/wsj_22-24.conll", worddict, chardict, tagdict)
    info("# train sentences: $(length(train_w))")
    info("# test sentences: $(length(test_w))")
    info("# words: $(length(worddict))")
    info("# chars: $(length(chardict))")
    info("# postags: $(length(tagdict))")

    nn = setup_nn(length(tagdict))
    opt = SGD()
    for epoch = 1:10
        println("epoch: $epoch")
        totalloss = 0.0
        batchsize = 1
        opt.rate = 0.0075 * sqrt(batchsize)/batchsize / epoch
        #opt.rate = 0.0075 / epoch
        batches = makebatch(batchsize, train_w, train_c, train_t)

        progress = Progress(length(batches[1]))
        for (w,c,t) in zip(batches...)
            y = nn(w, c)
            loss = crossentropy(t, y)
            totalloss += sum(loss.data)
            minimize!(opt, loss)
            next!(progress)
        end
        totalloss = round(totalloss/length(train_w), 5)
        println("loss: $(totalloss)")

        ys = Int[]
        zs = Int[]
        batches = makebatch(100, test_w, test_c, test_t)
        for (w,c,t) in zip(batches...)
            append!(ys, t.data)
            y = nn(w, c)
            z = argmax(y.data, 1)
            append!(zs, z)
        end
        length(ys) == length(zs) || throw("Length mismatch.")

        #ys = cat(1, map(x -> vec(x.data), test_t)...)
        #zs = cat(1, map(x -> vec(model(x).data), test_x)...)
        acc = mean(i -> ys[i] == zs[i] ? 1.0 : 0.0, 1:length(ys))
        acc = round(acc, 5)
        println("test acc.: $acc")
        println()
    end
end

include("data.jl")
include("model.jl")
train()
