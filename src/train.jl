function fit(init::Function, forward::Function, train_data, test_data, num_epochs::Int)
  for epoch = 1:num_epochs
    loss = 0.0
    println("epoch: $(epoch)")
    init(epoch)

    for i in randperm(length(train_data))
      x = train_data[i]
      y = forward(x)
      loss = ()
      gradient!(loss)
      update!()
    end
  end
end

type Model
  nn::Functor
  lossfun::Functor
  opt
end

function train()
  # model
  model = Model(path)
  opt = SGD(0.0075)

  for iter = 1:10
    println("iter: $(iter)")
    golds, preds = Int[], Int[]
    opt.learnrate = 0.0075 / iter
    loss = 0.0

    for i in randperm(length(traindata))
      tokens = traindata[i]
      append!(golds, map(t -> t.tagid, tokens))

      # forward & prediction
      out = forward(model, tokens)
      maxidx = maxrows(out.value)
      append!(preds, maxidx)

      # loss function
      p = map(t -> t.tagid, tokens)
      out = CrossEntropy(p)(out)
      loss += sum(out.value)

      # backward & update
      gradient!(out)
      update!(model, opt)
    end
    println("loss: $(loss)")
    acc = accuracy(golds, preds)
    println("train acc: $(acc)")
    decode(model, testdata)
    println("")
  end
  println("finish")
end
