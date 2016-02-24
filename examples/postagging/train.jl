using Merlin

function maxrows(m::Matrix)
  _, inds = findmax(m, 1)
  map!(i -> ind2sub(size(m), i)[1], inds)
  vec(inds)
end

function train(path)
  # data load
  worddict = read_wordlist("$(path)/words.lst")
  chardict, catdict = Dict(), Dict()
  traindata = read_conll("$(path)/wsj_00-18.conll", true, worddict, chardict, catdict)
  println("#word: $(length(worddict)), #char: $(length(chardict)), #cat: $(length(catdict))")
  #traindata = traindata[1:5000]
  testdata = read_conll("$(path)/wsj_22-24.conll", false, worddict, chardict, catdict)

  # model
  model = POSModel(path)
  opt = SGD(0.0075)

  for iter = 1:10
    println("iter: $(iter)")
    golds, preds = Int[], Int[]
    opt.learnrate = 0.0075 / iter
    loss = 0.0

    for i in randperm(length(traindata))
      tokens = traindata[i]
      append!(golds, map(t -> t.catid, tokens))

      out = forward(model, tokens)
      maxidx = maxrows(out.value)
      append!(preds, maxidx)

      # loss function
      px = zeros(Float32, 45, length(tokens))
      for j = 1:length(tokens)
        px[tokens[j].catid, j] = 1.0
      end
      out = CrossEntropy(px)(out)
      loss += sum(out.value)

      backward!(out)
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

function decode(m::POSModel, data::Vector{Vector{Token}})
  golds, preds = Int[], Int[]
  for i = 1:length(data)
    tokens = data[i]
    append!(golds, map(t -> t.catid, tokens))
    out = forward(m, tokens)
    append!(preds, maxrows(out.value))
  end
  acc = accuracy(golds, preds)
  println("test acc: $(acc)")
end
