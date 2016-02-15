using Merlin
using ArrayFire

type POSModel2
  wordlookup
  charlookup
  char_l
  sent_l
  sent_l2
end

function POSModel2(path)
  T = Float32
  wordlookup = Lookup(T, 100, 50000)
  charlookup = Lookup(T, 10, 100)
  charfun = Linear(T,50,50)
  sentfun = Linear(T,750,300)
  sentfun2 = Linear(T,300,45)
  POSModel2(wordlookup, charlookup, charfun, sentfun, sentfun2)
end

function forward(m::POSModel2, tokens::Vector{Token})
  wordvec = map(t -> t.wordid, tokens)
  wordmat = Identity(wordvec) |> m.wordlookup
  charvecs = map(tokens) do t
    y = Identity(t.charids) |> m.charlookup
    y = Window2D(y, 10, 5, 1, 1, 0, 2)
    y = m.char_l(y)
    y = Window2D(y, 1, length(t.charids), 1, 1)
    y = MaxPool2D(y, 1)
    y = Reshape(y, 50, 1)
    y
  end
  charmat = Concat(charvecs, 2)
  y = Concat([wordmat,charmat], 1)
  y = Window2D(y, 150, 5, 1, 1, 0, 2)
  y = m.sent_l(y)
  y = ReLU(y)
  y = m.sent_l2(y)
  y
end

type POSModel
  wordlookup
  charlookup
  charfun
  sentfun
end

function POSModel(path)
  T = Float32
  wordlookup = Lookup(T, 100, 50000)
  #wordlookup = Lookup("$(path)/nyt100.lst", T)
  charlookup = Lookup(T, 10, 100)
  charfun = [Window2D(10,5,1,1,0,2), Linear(T,50,50), Window2D(1,-1,1,1), MaxPool2D(1), Reshape(50,1)]
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

function optimize!( m::POSModel, opt::Optimizer)
  for f in [m.wordlookup, m.charlookup,m.charfun,m.sentfun]
    Merlin.optimize!(opt, f)
  end
end
