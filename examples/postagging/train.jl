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
  worddict = begin
    d = Dict()
    for l in open(readlines, "$(path)/words.lst")
      get!(d, chomp(l), length(d)+1)
    end
    d
  end
  chardict, catdict = Dict(), Dict()
  traindata = read_conll("$(path)/wsj_00-18.conll", true, worddict, chardict, catdict)
  println("#word: $(length(worddict)), #char: $(length(chardict)), #cat: $(length(catdict))")
  traindata = traindata[1:5000]
  testdata = read_conll("$(path)/wsj_22-24.conll", false, worddict, chardict, catdict)
  model = POSModel2(path)
  opt = SGD(0.0075)

  for iter = 1:10
    println("iter: $(iter)")
    golds, preds = Token[], Int[]
    opt.learnrate = 0.0075 / iter
    loss = 0.0

    #for i in randperm(length(traindata))
    for i = 1:length(traindata)
      #i % 100 == 0 && println(i)
      toks = traindata[i]
      append!(golds, toks)

      pred_var = forward(model, toks)
      #pred_device = maximum(pred_var.value, 1)
      #pred = to_host(pred_device)

      #target = onehot(45, map(t -> t.catid, toks), 1.0f0)

      #gold_var = map(t -> t.catid, toks) |> Variable
      #[gold_var, pred_var] |> CrossEntropy()
      #pred = maximum(var.value) |> to_host
      #append!(preds, pred)
      #tagids = map(t -> t.catid, toks)
      #target = zeros(var.value)
      #for j = 1:length(tagids)
      #  target[tagids[j], j] = 1.0
      #end

      #var = [Variable(target), var] |> CrossEntropy()
      #loss += sum(var.value)
      #var.grad = ones(var.value)
      #backward!(var)
      #optimize!(model, opt)
    end
    println("loss: $(loss)")
    #acc = eval(golds, preds)
    #println("train acc: $(acc)")
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
