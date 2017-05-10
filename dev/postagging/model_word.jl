using Merlin

function wordmodel()
  T = Float32
  embeds = Lookup(T,500000,100)
  @graph begin
    x = Var(:x)
    x = lookup(embeds, x)
    x = window(x, (100,5))
    x = Linear(T,500,300)(x)
    x = relu(x)
    x = Linear(T,300,45)(x)
    x
  end
end

#=
type WordModel
  sent_f
end

function Model(path::AbstractString)
  T = Float32
  embeds = Lookup(T,500000,100)
  sent_f = @graph begin
    x = Var(:x)
    x = lookup(embeds, x)
    x = window(x, (100,5))
    x = Linear(T, 500, 300)(x)
    x = relu(x)
    x = Linear(T, 300, 45)
    x
    #Window2D(100,5,1,1,0,2),
    #Linear(T,500,300),
    #ReLU(),
    #Linear(T,300,45)
  end
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
=#
