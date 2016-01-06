using Merlin

function create()
  chars = Variable()
  charsvec = chars |> Lookup(Char, T, 10) |> Window2D((10,5),(1,1)) |> Linear(T,50,50) |> MaxPool2D((1,-1),(1,1))
  charsvec |> MapReduce(Concat(2))
  words = Variable()
  wordmat = words |> Lookup(UTF8String, T, 100)
  sentmat = [charsvec, wordmat] |> Concat(1) |> Window2D((150,5),(1,1)) |> Linear(T,750,300) |> ReLU() |> Linear(T,300,45)
  compile(sentmat)
end

type POSModel
  wordembed
  charembed
  charfun
  sentfun
end

function POSModel(path)
  T = Float32
  #wordembed = Dict{UTF8String,Variable}()
  #charembed = Dict{Char,Variable}()
  wordembed = Lookup(UTF8String, T, 100)
  charembed = Lookup(Char, T, 10)
  charfun = [Window2D((-1,5),(1,1)), Linear(T,50,50), MaxPool2D((1,-1),(1,1))] |> Sequence
  sentfun = [Concat(1), Window2D((-1,5),(1,1)), Linear(T,750,300), ReLU(), Linear(T,300,45)] |> Sequence
  POSModel(wordembed, charembed, [charfun], sentfun)
end

function forward(m::POSModel, tokens::Vector{Token})
  ref = tokens[1]
  padt = Token(ref.dicts, "PADDING", [' '], -1)
  tokens = [padt; padt; tokens...; padt; padt]

  words = map(t -> t.word, tokens)
  m.wordembed(words)
  n_w = Node(m.wordembed)

  i = 1
  charvecs = map(tokens) do t
    chars = [' '; ' '; t.chars...; ' '; ' ']
    m.charembed(chars)
    n_c = Node(m.charembed)
    i > length(m.charfun) && push!(m.charfun, clone(m.charfun[1]))
    n_c = Node(m.charfun[i], n_c)
    i += 1
    n_c
  end
  n_c = Node(Concat(2), charvecs)
  n = Node(m.sentfun, [n_w, n_c])
  #n = Node(Concat(1), [n_w, n_c])
  #n = Node(m.sentfun, n)
  n
  #words = map(t -> t.word, tokens)
  #wordmat = m.wordembed(words)
  #charvecs = map(tokens) do t
  #  chars = [' '; ' '; t.chars...; ' '; ' ']
  #  chars |> m.charembed |> m.charfun
  #end
  #charmat = charvecs |> Concat(2)
  #[wordmat, charmat] |> Concat(1) |> m.sentfun
end
