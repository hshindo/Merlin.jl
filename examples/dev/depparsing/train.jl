function train(traindata::Vector, testdata::Vector)
    for epoch = 1:20
        println("epoch: $(epoch)")
        loss = 0.0
        for i in randperm(length(traindata))
            #for x in traindata
            x = traindata[i]
            sy = State(x, scorefun)
            sz = State(x, scorefun)
            y = beamsearch(sy, 1, expand_gold)[end][1]
            z = beamsearch(sz, 1, expand_pred)[end][1]
            loss += z.score - y.score
            max_violation!(y, z, train_gold, train_pred)
        end
        println("loss: $(loss)")
        golds, preds = Int[], Int[]
        for x in testdata
            s = State(x, scorefun)
            z = beamsearch(s, 1, expand_pred)[end][1]
            append!(golds, map(t -> t.headid, x))
            append!(preds, toheads(z))
        end
        acc = accuracy(golds, preds)
        println("test acc: $(acc)")
        println("")
    end
end
