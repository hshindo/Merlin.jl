type Graph <: Functor
  funs::Vector{Functor}
  tails::Vector{Vector{Int}}
end

Graph() = Graph(Functor[], Vector{Int}[])

function Base.push!(g::Graph, tails::Vector{Int}, funs::Functor...)
  ids = tails
  for f in funs
    push!(g.funs, f)
    push!(g.tails, ids)
    ids = [length(g.funs)]
  end
  ids[1]
end
Base.push!(g::Graph, tail::Int, funs::Functor...) = push!(g, [tail], funs...)
Base.push!(g::Graph, funs::Functor...) = push!(g, Int[], funs...)

function sequencial(funs::Functor...)
  g = Graph()
  push!(g, funs...)
  g
end

function apply(g::Graph, vars::Vector{Variable})
  outputs = Array(Any, length(g.funs))
  k = 1
  for i = 1:length(g.funs)
    f, tails = g.funs[i], g.tails[i]
    if length(tails) == 0
      outputs[i] = apply(f, vars[k])
      k += 1
    elseif length(tails) == 1
      outputs[i] = apply(f, outputs[tails[1]])
    else
      input = map(id -> outputs[id], tails)
      outputs[i] = apply(f, input)
    end
  end
  Variable(outputs[end], outputs)
end
