using Merlin

#function posmodel(path)
#  T = Float32
#  g = Graph()
#  wordfun = Lookup(UTF8String, T, 100)
#  charfun = Sequence(Lookup(Char, T, 10), Window1D(50, 10, 0), Linear(T, 50, 50), Pooling())
#  charfun = MapReduce(charfun, Concat(2))
#  w = push!(g, wordfun)
#  c = push!(g, charfun)
#  push!(g, [w, c], Concat(1), Window1D(750, 150, 0), Linear(T, 750, 300), ReLU(), Linear(T, 300, 45))
#  g
#end

type POSModel
  wordfun
  charfun
  sentfun
end

function POSModel(path)
  T = Float32
  #wordfun = Lookup(UTF8String, T, 100)
  wordfun = Lookup(path, UTF8String, T)
  charfun = Functor[Lookup(Char, T, 10), Window1D(50, 10, 0), Linear(T, 50, 50), Pooling()]
  sentfun = Functor[Window1D(750, 150, 0), Linear(T, 750, 300), ReLU(), Linear(T, 300, 45)]
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
