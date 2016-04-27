using Merlin

function train(path)
  # data
  worddict = read_wordlist("$(path)/words.lst")
  chardict, tagdict = Dict(), Dict()
  traindata = read_conll("$(path)/wsj_00-18.conll", true, worddict, chardict, tagdict)
  println("#word: $(length(worddict)), #char: $(length(chardict)), #tag: $(length(tagdict))")
  #traindata = traindata[1:10000]
  testdata = read_conll("$(path)/wsj_22-24.conll", false, worddict, chardict, tagdict)

  train_y = map(x -> map(t -> t.tagid, x), traindata)

  # model
  model = Model(path)
  #opt = SGD(0.0075)
  #nn(x::Vector{Token}) = forward(model, x)
  #function toloss(y::Vector{Token}, z::Variable)
  #  maxidx = argmax(z.value, 1)
  #  p = map(t -> t.tagid, y)
  #  CrossEntropy(p)(z)
  #end
  #t = Trainer(nn, toloss, opt)

  for epoch = 1:5
    println("epoch: $(epoch)")
    model.opt.rate = 0.0075 / epoch
    loss = fit(model, traindata, train_y)
    println("training loss: $(loss)")
    decode(model, testdata)
    println("")
  end
  #=
  for iter = 1:10
    println("iter: $(iter)")
    opt.rate = 0.0075 / iter
    loss = 0.0

    for i in randperm(length(traindata))
      tokens = traindata[i]

      # forward & prediction
      out = forward(model, tokens)
      maxidx = argmax(out.value, 1)

      # loss function
      p = map(t -> t.tagid, tokens)
      out = CrossEntropy(p)(out)
      loss += sum(out.value)

      # backward & update
      #gradient!(out)
      #update!(model, opt)
    end
    println("loss: $(loss)")
    #acc = accuracy(golds, preds)
    #println("train acc: $(acc)")
    decode(model, testdata)
    println("")
  end
  =#
  println("finish")
end

function decode(m::Model, data::Vector{Vector{Token}})
  golds, preds = Int[], Int[]
  for i = 1:length(data)
    tokens = data[i]
    append!(golds, map(t -> t.tagid, tokens))
    out = forward(m, tokens)
    append!(preds, argmax(out.value,1))
  end
  acc = accuracy(golds, preds)
  println("test acc: $(acc)")
end
