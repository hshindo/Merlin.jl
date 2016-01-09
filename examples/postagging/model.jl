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
  #wordembed = Lookup("$(path)/nyt.100", UTF8String, T)
  charembed = Lookup(Char, T, 10)
  charfun = [Window2D((-1,5),(1,1)), Linear(T,50,50), MaxPool2D((1,-1),(1,1))]
  sentfun = [Window2D((150,5),(1,1)), Linear(T,750,300), ReLU(), Linear(T,300,45)]
  POSModel(wordembed, charembed, charfun, sentfun)
end

function forward(m::POSModel, tokens::Vector{Token})
  ref = tokens[1]
  padt = Token(ref.dicts, "PADDING", [' '], -1)
  tokens = [padt; padt; tokens...; padt; padt]

  words = map(t -> t.word, tokens)
  wordmat = Variable(words) |> m.wordembed
  charvecs = map(tokens) do t
    chars = [' '; ' '; t.chars...; ' '; ' ']
    Variable(chars) |> m.charembed |> m.charfun
  end
  charmat = charvecs |> Concat(2)
  [wordmat, charmat] |> Concat(1) |> m.sentfun
end

function optimize!( m::POSModel, opt::Optimizer)
  Merlin.optimize!(opt, m.wordembed)
  Merlin.optimize!(opt, m.charembed)
  Merlin.optimize!(opt, m.charfun)
  Merlin.optimize!(opt, m.sentfun)
end
