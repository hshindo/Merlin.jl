using HDF5

type Segmenter
    word2id::Dict
    char2id::Dict
    tag2id::Dict
    id2tag::Dict
    model
end

function Segmenter()
    h5file = "glove.6B.100d.h5"
    words = h5read(h5file, "key")
    push!(words, "UNKNOWN")
    word2id = Dict(words[i] => i for i=1:length(words))
    char2id = Dict{String,Int}()
    tag2id = Dict{String,Int}()
    id2tag = Dict{Int,String}()
    wordembeds = h5read(h5file, "value")
    wordembeds = cat(2, wordembeds, uniform(Float32,-0.001,0.001,100,1))
    charembeds = rand(Float32, 50, 100)
    model = Model(wordembeds, charembeds, 13)
    Segmenter(word2id, char2id, tag2id, id2tag, model)
end

function train(seg::Segmenter, trainfile::String, testfile::String)
    train_x, train_y = read!(seg, trainfile)
    test_x, test_y = read!(seg, testfile)
    info("# sentences of train data: $(length(train_x))")
    info("# sentences of test data: $(length(test_x))")
    info("# words: $(length(seg.word2id))")
    info("# chars: $(length(seg.char2id))")
    info("# tags: $(length(seg.tag2id))")

    opt = SGD(0.005)
    for epoch = 1:20
        println("epoch: $epoch")
        #opt.rate = 0.0075 / epoch
        loss = fit(train_x, train_y, seg.model, opt)
        println("loss: $loss")

        ys = cat(1, map(x -> vec(x.data), test_y)...)
        zs = cat(1, map(x -> vec(seg.model(x).data), test_x)...)
        acc = mean(i -> ys[i] == zs[i] ? 1.0 : 0.0, 1:length(ys))
        acc = round(acc, 5)
        preds = map(id -> seg.id2tag[id], zs)
        golds = map(id -> seg.id2tag[id], ys)
        f = fscore(preds, golds)
        println("test eval...")
        println("acc: $acc")
        println("prec: $(f[1])")
        println("recall: $(f[2])")
        println("fscore: $(f[3])")
        println()
    end
end

function read!(seg::Segmenter, path::String)
    data_x, data_y = Tuple{Var,Vector{Var}}[], Var[]
    w, c, t = Int[], Vector{Int}[], Int[]
    unkword = seg.word2id["UNKNOWN"]
    lines = open(readlines, path)
    for i = 1:length(lines)
        line = chomp(lines[i])
        if isempty(line) || i == length(lines)
            isempty(w) && continue
            w = Var(reshape(w,1,length(w)))
            c = map(x -> Var(reshape(x,1,length(x))), c)
            t = Var(reshape(t,1,length(t)))
            push!(data_x, (w,c))
            push!(data_y, t)
            w, c, t = Int[], Vector{Int}[], Int[]
        else
            items = split(line, "\t")
            word = String(items[1])
            #word0 = replace(word, r"[0-9]", '0')
            word0 = word
            wordid = get(seg.word2id, lowercase(word0), unkword)
            push!(w, wordid)

            chars = Vector{Char}(word0)
            charids = map(c -> get!(seg.char2id,string(c),length(seg.char2id)+1), chars)
            push!(c, charids)

            tag = String(items[3])
            tagid = get!(seg.tag2id, tag, length(seg.tag2id)+1)
            seg.id2tag[tagid] = tag
            push!(t, tagid)
        end
    end
    data_x, data_y
end

include("eval.jl")
include("model.jl")

seg = Segmenter()
#path = joinpath(dirname(@__FILE__), ".data")
#train(seg, "$(path)/eng.train", "$(path)/eng.testb")

# chunking
path = joinpath(dirname(@__FILE__), ".data/chunking")
train(seg, "$(path)/train.txt", "$(path)/test.txt")
