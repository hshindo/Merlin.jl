using Merlin

type Model
  sent_f
end

function Model(path::AbstractString)
  T = Float32
  sent_f = [Lookup(T,500000,100),
            #Lookup("$(path)/nyt100.lst", T),
            Window2D(100,5,1,1,0,2),
            Linear(T,500,300),
            ReLU(),
            Linear(T,300,45)]
  Model(sent_f)
end

function forward(m::Model, tokens::Vector{Token})
  wordvec = map(t -> t.wordid, tokens)
  wordvec = reshape(wordvec, 1, length(wordvec)) |> Variable
  wordmat = m.sent_f(wordvec)
end

function update!(m::Model, opt)
  Merlin.update!(opt, m.sent_f)
end
