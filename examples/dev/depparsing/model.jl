type Model
    ws::Vector
end

function (m::Model)(s::State)
    tokens = s.tokens
    s0 = tokens[s.top]
    s1 = isnull(s.left) ? tokens[s.left.top] : nulltoken
    b0 = s.right <= length(tokens) ? tokens[s.right] : nulltoken
    sl = isnull(s.lch) ? tokens[s.lch.top] : nulltoken
    sr = isnull(s.rch) ? tokens[s.rch.top] : nulltoken
    
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
