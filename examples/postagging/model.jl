using Merlin

function posmodel(path)
  T = Float32
  g = Graph()
  wordembed = Lookup(UTF8String, T, 100)
  charembed = Lookup(Char, T, 10)
  charfun = sequencial(charembed, Window1D(50, 10, 0), Linear(T, 50, 50), Pooling())
  w = push!(g, wordembed)
  c = push!(g, MapReduce(charfun, Concat(2)))
  push!(g, [w, c], Concat(1), Window1D(750, 150, 0), Linear(T, 750, 300), ReLU(), Linear(T, 300, 45))
  g
end

type POSModel
  wordembed
  charseq
  sentseq
end

function POSModel(path)
  T = Float32
  wordembed = Lookup(UTF8String, T, 100)
  charseq = Functor[Lookup(Char, T, 10), Window1D(50, 10, 0), Linear(T, 50, 50), Pooling()]
  charsseq = [wordembed, Map(charseq)] |> Concat(1)
  sentseq = Functor[Window1D(750, 150, 0), Linear(T, 750, 300), ReLU(), Linear(T, 300, 45)]
  POSModel(wordembed, charseq, sentseq)
end

function forward(m::POSModel, tokens::Vector{Token})
  ref = tokens[1]
  padt = Token(ref.dicts, "PADDING", [' '], -1)
  tokens = [padt; padt; tokens...; padt; padt]
  words = map(t -> t.word, tokens)
  wordmat = Node(words) |> m.wordembed
  charmat = map(tokens) do t
    chars = [' '; ' '; t.chars...; ' '; ' ']
    Node(chars) |> m.charseq
  end
  charmat = charmat |> Concat(2)
  [wordmat, charmat] |> Concat(1) |> m.sentseq
end

type LayerSet
end
