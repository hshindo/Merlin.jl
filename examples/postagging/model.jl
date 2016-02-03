using Merlin

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
  charfun = [Window2D(10,5,1,1,0,2), Linear(T,50,50), Window2D(1,-1,1,1), MaxPool2D()]
  sentfun = [Window2D(150,5,1,1,0,2), Linear(T,750,300), ReLU(), Linear(T,300,45)]
  POSModel(wordlookup, charlookup, charfun, sentfun)
end

function forward(m::POSModel, tokens::Vector{Token})
  wordvec = map(t -> t.wordid, tokens) |> AFArray
  wordmat = Variable(wordvec) |> m.wordlookup
  charvecs = map(tokens) do t
    idx = AFArray(t.charids)
    a = Variable(idx) |> m.charlookup
    println("ok1")
    b = a |> m.charfun
    println("ok2")
    b
  end
  println("ok3")
  error("")
  charmat = charvecs |> Concat(2)
  [wordmat, charmat] |> Concat(1) |> m.sentfun
end

function optimize!( m::POSModel, opt::Optimizer)
  Merlin.optimize!(opt, m.wordlookup)
  Merlin.optimize!(opt, m.charlookup)
  Merlin.optimize!(opt, m.charfun)
  Merlin.optimize!(opt, m.sentfun)
end
