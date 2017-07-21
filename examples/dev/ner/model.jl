type Model
    fw
    fc
    fs
end

function Model(wordembeds::Matrix{T}, ntags::Int) where T
    fw = @graph w begin
        Lookup(wordembeds)(w)
    end
    fc = @graph c begin
        c = Lookup(T,100,10)(c)
        c = Conv1D(T,50,50,20,10)(c)
        max(c, 2)
    end
    fs = @graph s begin
        s = Conv1D(T,750,300,300,150)(s)
        s = relu(s)
        Linear(T,300,ntags)(s)
    end
    Model(fw, fc, fs)
end

function (m::Model)(word::Var, chars::Vector{Var})
    w = m.fw(word)
    cs = Var[]
    for i = 1:length(chars)
        push!(cs, m.fc(chars[i]))
    end
    c = cat(2, cs...)
    s = cat(1, w, c)
    m.fs(s)
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
