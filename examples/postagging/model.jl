using Merlin

type POSModel
  word_f
  char_f
  sent_f
end

function POSModel(path)
  T = Float32
  #word_f = Lookup(T, 500000, 100)
  word_f = Lookup("$(path)/nyt100.lst", T)
  char_f = [Lookup(T,100,10),
            Window2D(10,5,1,1,0,2),
            Linear(T,50,50),
            Max(2)]
  sent_f = [Window2D(150,5,1,1,0,2),
            Linear(T,750,300),
            ReLU(),
            Linear(T,300,45)]
  POSModel(word_f, char_f, sent_f)
end

function forward(m::POSModel, tokens::Vector{Token})
  wordvec = map(t -> t.wordid, tokens) |> Variable
  wordmat = m.word_f(wordvec)
  charvecs = map(tokens) do t
    v = Variable(t.charids)
    m.char_f(v)
  end
  charmat = charvecs |> Concat(2)
  [wordmat, charmat] |> Concat(1) |> m.sent_f
end

function update!(m::POSModel, opt::Optimizer)
  for f in (m.word_f, m.char_f, m.sent_f)
    Merlin.update!(opt, f)
  end
end
