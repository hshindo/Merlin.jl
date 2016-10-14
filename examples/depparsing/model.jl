function model()
    
end

function train()
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
