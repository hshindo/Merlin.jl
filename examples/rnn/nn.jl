using Merlin

type Token
  word::AbstractString
  wordid::Int
end

type Model
  word_f::Functor
  rnn::Functor
  h0::Vector # initial hidden vector
end

function Model()
  T = Float32
  word_f = Lookup(F, 10000, 100)
  rnn = GRU(T, 100)
  h0 = ones(T, 100, 1)
  Model(word_f, rnn, h0)
end

function forward{T}(tokens::Vector{Token}, m::Model)
  h = m.h0
  for i = 1:length(tokens)
    wordvec = word_f([tokens[i].wordid])
    h = m.rnn(wordvec, h)
  end
  h
end
