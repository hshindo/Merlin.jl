using Merlin
using ArrayFire

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
  catdictt = begin

  end
  traindata = readCoNLL("$(path)/wsj_00-18.conll", catdict)
  traindata = traindata[1:5000]
  testdata = readCoNLL("$(path)/wsj_22-24.conll", catdict)
  model = POSModel(path)
  opt = SGD(0.0075)

  for iter = 1:10
    println("iter: $(iter)")
    golds, preds = Token[], Int[]
    opt.learnrate = 0.0075 / iter
    loss = 0.0
    for i in randperm(length(traindata))
      toks = traindata[i]
      append!(golds, toks)

      var = forward(model, toks)
      append!(preds, maxrows(var.value))
      tagids = map(t -> t.catid, toks)
      target = zeros(var.value)
      for j = 1:length(tagids)
        target[tagids[j], j] = 1.0
      end

      var = [Variable(target), var] |> CrossEntropy()
      loss += sum(var.value)
      #var.grad = ones(var.value)
      #backward!(var)
      #optimize!(model, opt)
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
    var = forward(m, toks)
    append!(preds, maxrows(var.value))
  end
  acc = eval(golds, preds)
  println("test acc: $(acc)")
end
