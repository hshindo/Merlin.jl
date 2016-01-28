using Merlin

type POSModel
  wordlookup
  charlookup
  charfun
  sentfun
end

function POSModel(path)
  T = Float32
  wordlookup = AFLookup(T, 50000, 100)
  #wordlookup = Lookup("$(path)/nyt100.lst", T)
  charlookup = AFLookup(T, 100, 10)
  charfun = [Window2D((10,5),(1,1),(0,2)), Linear(T,50,50), MaxPool2D((1,-1),(1,1))]
  sentfun = [Window2D((150,5),(1,1),(0,2)), Linear(T,750,300), ReLU(), Linear(T,300,45)]
  POSModel(wordlookup, charlookup, charfun, sentfun)
end

function forward(m::POSModel, tokens::Vector{Token})
  #ref = tokens[1]
  #padt = Token(ref.dicts, "PADDING", [' '], -1)
  #tokens = [padt; padt; tokens...; padt; padt]

  wordvec = map(t -> t.wordid, tokens)
  wordmat = Variable(wordvec) |> m.wordlookup
  charvecs = map(tokens) do t
    #chars = [' '; ' '; t.chars...; ' '; ' ']
    Variable(t.charids) |> m.charlookup |> m.charfun
  end
  charmat = charvecs |> Concat(2)
  [wordmat, charmat] |> Concat(1) |> m.sentfun
end

function optimize!( m::POSModel, opt::Optimizer)
  Merlin.optimize!(opt, m.wordlookup)
  Merlin.optimize!(opt, m.charlookup)
  Merlin.optimize!(opt, m.charfun)
  Merlin.optimize!(opt, m.sentfun)
end
