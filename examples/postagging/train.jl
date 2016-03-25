using Merlin

function maxrows(m::Matrix)
  _, inds = findmax(m, 1)
  map!(i -> ind2sub(size(m), i)[1], inds)
  vec(inds)
end

function train(path)
  # data
  worddict = read_wordlist("$(path)/words.lst")
  chardict, tagdict = Dict(), Dict()
  traindata = read_conll("$(path)/wsj_00-18.conll", true, worddict, chardict, tagdict)
  println("#word: $(length(worddict)), #char: $(length(chardict)), #tag: $(length(tagdict))")
  traindata = traindata[1:10000]
  testdata = read_conll("$(path)/wsj_22-24.conll", false, worddict, chardict, tagdict)

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
    #decode(model, testdata)
    println("")
  end
  println("finish")
end

function decode(m::Model, data::Vector{Vector{Token}})
  golds, preds = Int[], Int[]
  for i = 1:length(data)
    tokens = data[i]
    append!(golds, map(t -> t.tagid, tokens))
    out = forward(m, tokens)
    append!(preds, maxrows(out.value))
  end
  acc = accuracy(golds, preds)
  println("test acc: $(acc)")
end
