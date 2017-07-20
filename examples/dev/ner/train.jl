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
    # h5file = "glove.6B.100d.h5"
    wordembeds_file = "wordembeds_nyt100.h5"
    words = h5read(wordembeds_file, "s")
    word2id = Dict(words[i] => i for i=1:length(words))
    char2id = Dict{String,Int}()
    tag2id = Dict{String,Int}()
    id2tag = Dict{Int,String}()
    wordembeds = h5read(wordembeds_file, "v")
    charembeds = rand(Float32, 100, 100)
    #nn = Model(wordembeds, charembeds, ntags)
    nn = nothing
    Segmenter(word2id, char2id, tag2id, id2tag, nn)
end

function train(seg::Segmenter, trainfile::String, testfile::String)
    train_w, train_c, train_t = readdata!(seg, trainfile)
    test_w, test_c, test_t = readdata!(seg, testfile)
    info("# sentences of train data: $(length(train_w))")
    info("# sentences of test data: $(length(test_w))")
    info("# words: $(length(seg.word2id))")
    info("# chars: $(length(seg.char2id))")
    info("# tags: $(length(seg.tag2id))")

    opt = SGD(0.005)
    for epoch = 1:1
        println("epoch: $epoch")
        #opt.rate = 0.0075 / epoch

        function train_f(data::Tuple)
            w, c, t = data
            y = seg.nn(w, c)
            softmax_crossentropy(t, y)
        end
        train_data = collect(zip(train_w, train_c, train_t))
        loss = minimize!(train_f, opt, train_data)
        println("loss: $loss")

        # test
        println("testing...")
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
        println("acc: $acc")
        println("prec: $prec")
        println("recall: $recall")
        println("fscore: $fval")
        println()
    end
end

include("data.jl")
include("eval.jl")
include("model.jl")

# training
seg = Segmenter(13)
path = joinpath(dirname(@__FILE__), ".data")
#train(seg, "$(path)/eng.train", "$(path)/eng.testb")

Merlin.save("ner.h5", seg)
#using Merlin
#Merlin.load(joinpath(dirname(@__FILE__),"ner.h5"))

# chunking
# path = joinpath(dirname(@__FILE__), ".data/chunking")
# train(seg, "$(path)/train.txt", "$(path)/test.txt")
