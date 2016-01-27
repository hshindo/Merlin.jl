using Merlin

type POSModel
  wordlookup
  charlookup
  charfun
  sentfun
end

function POSModel(path)
  T = Float32
  wordlookup = begin
    keys = map(l -> UTF8String(chomp(l)), open(readlines,"$(path)/word.lst"))
    Lookup(keys, T, 100)
  end
  charlookup = begin
    keys = map(l -> Char(chomp(l)), open(readlines,"$(path)/char.lst"))
    Lookup(keys, T, 10)
  end
  charfun = [Window2D((10,5),(1,1),(0,2)), Linear(T,50,50), MaxPool2D((1,-1),(1,1))]
  sentfun = [Window2D((150,5),(1,1),(0,2)), Linear(T,750,300), ReLU(), Linear(T,300,45)]
  POSModel(wordlookup, charlookup, charfun, sentfun)
end

function forward(m::POSModel, tokens::Vector{Token})
  #ref = tokens[1]
  #padt = Token(ref.dicts, "PADDING", [' '], -1)
  #tokens = [padt; padt; tokens...; padt; padt]

  words = map(t -> t.word, tokens)
  wordmat = Variable(words) |> m.wordembed
  charvecs = map(tokens) do t
    #chars = [' '; ' '; t.chars...; ' '; ' ']
    chars = t.chars
    Variable(chars) |> m.charembed |> m.charfun
  end
  charmat = charvecs |> Concat(2)
  [wordmat, charmat] |> Concat(1) |> m.sentfun
end
