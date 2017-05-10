using Merlin
using Compat

type Model
  word_f
  char_f
  sent_f
end

function Model(path)
  T = Float32
  word_f = Lookup(T, 500000, 100)
  char_f = @graph begin
    x = Var(:x)
    x = Lookup(T,100,10)(x)
    Window2D(10,5,1,1,0,2)
    x = Linear(T,50,50)(x)
    x = max(x,2)
    x
  end
  sent_f = @graph begin
    Window2D(150,5,1,1,0,2)
    Linear(T,750,300)
    x = relu(x)
    x = Linear(T,300,45)(x)
    x
  end
  Model(word_f, char_f, sent_f)
end

@compat function (m::Model)(tokens::Vector{Token})
  wordvec = map(t -> t.wordid, tokens)
  wordvec = reshape(wordvec, 1, length(wordvec))
  wordmat = m.word_f(wordvec)
  charvecs = map(tokens) do t
    charvec = reshape(t.charids, 1, length(t.charids))
    m.char_f(charvec)
  end
  charmat = Concat(2)(charvecs)
  (wordmat, charmat) |> Concat(1) |> m.sent_f
end
