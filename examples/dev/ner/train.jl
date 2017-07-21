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
    charembeds = rand(Float32, 100, 100)
    nn = Model(wordembeds, charembeds, ntags)
    Segmenter(word2id, char2id, tag2id, id2tag, nn)
end

function train(seg::Segmenter, trainfile::String, testfile::String)
    train_w, train_c, train_t = readdata!(seg, trainfile)
    test_w, test_c, test_t = readdata!(seg, testfile)
    info("# Training sentences:\t$(length(train_w))")
    info("# Testing sentences:\t$(length(test_w))")
    info("# Words:\t$(length(seg.word2id))")
    info("# Chars:\t$(length(seg.char2id))")
    info("# Tags:\t$(length(seg.tag2id))")

    opt = SGD(0.005)
    for epoch = 1:1
        println("Epoch:\t$epoch")
        #opt.rate = 0.0075 / epoch

        function train_f(data::Tuple)
            w, c, t = data
            y = seg.nn(w, c)
            softmax_crossentropy(t, y)
        end
        train_data = collect(zip(train_w, train_c, train_t))
        loss = minimize!(train_f, opt, train_data)
        println("Loss:\t$loss")

        # test
        println("Testing...")
        function test_f(data::Tuple)
            w, c = data
            y = seg.nn(w, c)
            vec(argmax(y.data,1))
        end
        test_data = collect(zip(test_w, test_c))
        ys = cat(1, map(t -> t.data, test_t)...)
        zs = cat(1, map(test_f, test_data)...)
        length(ys) == length(zs) || throw("Length mismatch.")
        acc = mean(i -> ys[i] == zs[i] ? 1.0 : 0.0, 1:length(ys))
        acc = round(acc, 5)
        preds = map(id -> seg.id2tag[id], zs)
        golds = map(id -> seg.id2tag[id], ys)
        prec, recall, fval = fscore(preds, golds)
        println("Accuracy:\t$acc")
        println("Precision:\t$prec")
        println("Recall:\t$recall")
        println("FScore:\t$fval")
        println()
    end
end

include("data.jl")
include("eval.jl")
include("model.jl")

# training
seg = Segmenter(13)
path = joinpath(dirname(@__FILE__), ".data")
train(seg, "$(path)/eng.train", "$(path)/eng.testb")
Merlin.save("ner.h5", seg.nn)

#using Merlin
#Merlin.load(joinpath(dirname(@__FILE__),"ner.h5"))

# chunking
# path = joinpath(dirname(@__FILE__), ".data/chunking")
# train(seg, "$(path)/train.txt", "$(path)/test.txt")
