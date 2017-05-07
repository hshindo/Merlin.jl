type Segmenter
    worddict
    chardict
    tagdict
end

function Segmenter()
    wordembeds = h5read(h5file, "v")
    charembeds = rand(Float32, 10, length(chardict))
    model = Model(wordembeds, charembeds, length(tagdict))
end

function train(seg::Segmenter, trainfile::String, testfile::String)
    train_x, train_y = read!(seg, trainpath)
    test_x, test_y = read!(seg, testpath)
    info("# sentences of train data: $(length(train_x))")
    info("# sentences of test data: $(length(test_x))")
    info("# words: $(length(seg.worddict))")
    info("# chars: $(length(seg.chardict))")
    info("# tags: $(length(seg.tagdict))")

    opt = SGD()
    for epoch = 1:10
        println("epoch: $epoch")
        opt.rate = 0.0075 / epoch
        loss = fit(train_x, train_y, model, opt)
        println("loss: $loss")

        ys = cat(1, map(x -> vec(x.data), test_y)...)
        zs = cat(1, map(x -> vec(model(x).data), test_x)...)
        acc = mean(i -> ys[i] == zs[i] ? 1.0 : 0.0, 1:length(ys))
        preds = map(id -> tagdict[id], zs)
        golds = map(id -> tagdict[id], ys)
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
    unkword = worddict["UNKNOWN"]
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
            word0 = replace(word, r"[0-9]", '0')
            wordid = get(worddict, lowercase(word0), unkword)
            push!(w, wordid)

            chars = Vector{Char}(word0)
            charids = map(c -> get!(chardict,c,length(chardict)+1), chars)
            push!(c, charids)

            tag = String(items[2])
            tagid = get!(tagdict, tag, length(tagdict)+1)
            push!(t, tagid)
        end
    end
    data_x, data_y
end
