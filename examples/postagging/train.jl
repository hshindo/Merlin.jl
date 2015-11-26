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
  #ls = LayerSet(path)
  model = posmodel(path)
  opt = SGD(0.0075)

  for iter = 1:10
    println("iter: $(iter)")
    golds, preds = Token[], Int[]
    opt.learnrate = 0.0075 / iter
    loss = 0.0
    for i in randperm(length(traindata))
      toks = traindata[i]
      append!(golds, toks)

      ref = toks[1]
      padt = Token(ref.dicts, "PADDING", [' '], -1)
      padtoks = [padt; padt; toks...; padt; padt]
      words = map(t -> t.word, padtoks)
      chars = map(t -> [' ', ' ', t.chars..., ' ', ' '], padtoks)
      node = [Variable(words), Variable(chars)] |> model

      append!(preds, maxrows(node.data))
      tagids = map(t -> t.catid, toks)
      target = zeros(node.data)
      for j = 1:length(tagids)
        target[tagids[j], j] = 1.0
      end
      node = node |> CrossEntropy(target)
      loss += sum(node.data)
      #diff!(node)
      #optimize!(opt, ls.wordembed)
      #optimize!(opt, ls.charembed)
      #optimize!(opt, ls.l1)
      #optimize!(opt, ls.l2)
      #optimize!(opt, ls.l3)
    end
    println("loss: $(loss)")
    acc = eval(golds, preds)
    println("train acc: $(acc)")
    #decode(testdata, ls)
    println("")
  end
  println("finish")
end
