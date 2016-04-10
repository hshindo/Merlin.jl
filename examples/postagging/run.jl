ENV["USE_CUDA"] = true
workspace()
using Merlin

include("token.jl")

T = Float32
word_f = Lookup(T, 500000, 100)
#word_f = Lookup("$(path)/nyt100.lst", T)
char_f = Sequence(Lookup(T,100,10),
                  Window2D(10,5,1,1,0,2),
                  Linear(T,50,50),
                  Max(2))
sent_f = Sequence(Window2D(150,5,1,1,0,2),
                  Linear(T,750,300),
                  ReLU(),
                  Linear(T,300,45))

function forward(m::Model, tokens::Vector{Token})
  wordvec = map(t -> t.wordid, tokens)
  wordvec = reshape(wordvec, 1, length(wordvec)) |> Variable
  wordmat = m.word_f(wordvec)
  charvecs = map(tokens) do t
    charvec = reshape(t.charids, 1, length(t.charids)) |> Variable
    m.char_f(charvec)
  end
  charmat = charvecs |> Concat(2)
  [wordmat, charmat] |> Concat(1) |> m.sent_f
end

function update!(m::Model, opt)
  for f in (word_f, char_f, sent_f)
    Merlin.update!(opt, f)
  end
end






include("model_char.jl")
include("train.jl")
path = "C:/Users/hshindo/Dropbox/tagging"

@time train(path)
