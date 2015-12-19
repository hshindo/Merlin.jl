using Merlin

type POSModel
  wordfun
  charfun
  sentfun
end

function POSModel(path)
  T = Float32
  wordfun = Lookup(UTF8String, T, 100)
  #wordfun = Lookup(path, UTF8String, T)
  charfun = [Lookup(Char,T,10), Window2D((10,5),(1,1)), Linear(T,50,50), MaxPool2D((1,-1),(1,1))]
  sentfun = [Window2D((150,5),(1,1)), Linear(T,750,300), ReLU(), Linear(T,300,45)]
  POSModel(wordfun, charfun, sentfun)
end

function forward(m::POSModel, tokens::Vector{Token})
  ref = tokens[1]
  padt = Token(ref.dicts, "PADDING", [' '], -1)
  tokens = [padt; padt; tokens...; padt; padt]
  words = map(t -> t.word, tokens)
  wordmat = Variable(words) |> m.wordfun
  charvecs = map(tokens) do t
    chars = [' '; ' '; t.chars...; ' '; ' ']
    Variable(chars) |> m.charfun
  end
  charmat = charvecs |> Concat(2)
  [wordmat, charmat] |> Concat(1) |> m.sentfun
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
