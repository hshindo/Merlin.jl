using Merlin
using ArrayFire

type POSModel
  wordlookup
  charlookup
  charfun
  sentfun
end

function POSModel(path)
  T = Float32
  wordlookup = Lookup(T, 100, 500000)
  #wordlookup = Lookup("$(path)/nyt100.lst", T)
  charlookup = Lookup(T, 10, 100)
  charfun = [Window2D(10,5,1,1,0,2), Linear(T,50,50), Window2D(1,-1,1,1,0,0,false), MaxPooling(2)]
  sentfun = [Window2D(150,5,1,1,0,2), Linear(T,750,300), ReLU(), Linear(T,300,45)]
  POSModel(wordlookup, charlookup, charfun, sentfun)
end

function forward(m::POSModel, tokens::Vector{Token})
  wordvec = map(t -> t.wordid, tokens)
  wordmat = Variable(wordvec) |> m.wordlookup
  charvecs = map(tokens) do t
    Variable(t.charids) |> m.charlookup |> m.charfun
  end
  charmat = charvecs |> Concat(2)
  [wordmat, charmat] |> Concat(1) |> m.sentfun
end

function update!( m::POSModel, opt::Optimizer)
  for f in (m.wordlookup, m.charlookup, m.sentfun, m.charfun)
    applicable(optimize!, opt, f) && optimize!(opt, f)
  end
end
