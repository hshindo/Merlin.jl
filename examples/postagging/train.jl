using Merlin
using ArrayFire



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
  model = POSModel(path)
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

      #l = length(AF.af_ptrs)
      out = forward(model, toks)
      #m, idx = findmax(out.value, 1)
      #idx = convert(Vector{Int}, vec(to_host(idx)))
      #append!(preds, idx+1)

      px = zeros(Float32, 45, length(toks))
      for j = 1:length(toks)
        px[toks[j].catid, j] = -1.0f0
      end

      out = CrossEntropy(px)(out)
      #loss += sum(out.value)
      #p = device_ptr(out.value)
      #host = pointer_to_array(p, size(out.value))

      #l = sum(sum(out.value, 2), 1)
      #loss += to_host(l)[1]
      out.grad = ones(out.value)
      backward!(out)

      #target = onehot(45, map(t -> t.catid, toks), 1.0f0)

      #gold_var = map(t -> t.catid, toks) |> Variable
      #[gold_var, pred_var] |> CrossEntropy()
      #pred = maximum(var.value) |> to_host

      #tagids = map(t -> t.catid, toks)
      #target = zeros(var.value)
      #for j = 1:length(tagids)
      #  target[tagids[j], j] = 1.0
      #end

      #var = [Variable(target), var] |> CrossEntropy()
      #loss += sum(var.value)
      #var.grad = ones(var.value)
      #backward!(var)
      #update!(model, opt)
      #AF.free(l)
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
