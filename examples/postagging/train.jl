using Merlin

function maxrows(m::Matrix)
  _, inds = findmax(m, 1)
  map!(i -> ind2sub(size(m), i)[1], inds)
  vec(inds)
end

function makebatchs(data::Vector{Vector{Token}}, batchsize::Int)
  batchs = Vector{Token}[]
  pad = getpad(data[1][1])
  i = 1
  while i <= length(data)
    range = i:(i + + min(batchsize, length(data) - i + 1) - 1)
    push!(batchs, [pad; pad; data[range]...; pad; pad])
    i += batchsize
  end
  batchs
end

function train(path)
  dicts = (Dict(), Dict(), Dict())
  traindata = readCoNLL("$(path)/wsj_00-18.conll", dicts)
  traindata = traindata[1:5000]
  testdata = readCoNLL("$(path)/wsj_22-24.conll", dicts)
  model = POSModel("$(path)/nyt.100")
  opt = SGD(0.0075)

  for iter = 1:10
    println("iter: $(iter)")
    golds, preds = Token[], Int[]
    opt.learnrate = 0.0075 / iter
    loss = 0.0
    for i in randperm(length(traindata))
      toks = traindata[i]
      append!(golds, toks)

      #ref = toks[1]
      #padt = Token(ref.dicts, "PADDING", [' '], -1)
      #padtoks = [padt; padt; toks...; padt; padt]
      #w = map(t -> t.word, padtoks) |> Variable
      #c = map(t -> Variable([' ', ' ', t.chars..., ' ', ' ']), padtoks)
      #node = (w, c) |> model
      node = forward(model, toks)

      append!(preds, maxrows(node.value))
      tagids = map(t -> t.catid, toks)
      target = zeros(node.value)
      for j = 1:length(tagids)
        target[tagids[j], j] = 1.0
      end
      node = node |> CrossEntropy(target)
      loss += sum(node.value)
      node.grad = ones(node.value)
      backward!(node)
      optimize!(opt, model.wordembed)
      optimize!(opt, model.charembed)
      optimize!(opt, model.charfun)
      optimize!(opt, model.sentfun)
    end
    println("loss: $(loss)")
    acc = eval(golds, preds)
    println("train acc: $(acc)")
    #decode(model, testdata)
    println("")
  end
  println("finish")
end

function decode(m::POSModel, data::Vector{Vector{Token}})
  golds, preds = Token[], Int[]
  for i = 1:length(data)
    toks = data[i]
    append!(golds, toks)
    node = forward(m, toks)
    append!(preds, maxrows(node.value))
  end
  acc = eval(golds, preds)
  println("test acc: $(acc)")
end
