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

function apply(g::Graph, var::Variable)
  input = var.data
  l = filter(i -> length(g.tails[i]) == 0, 1:length(g.funs)) # must be rewrited
  outputs = Array(Any, length(g.funs))
  k = 1
  for i = 1:length(g.funs)
    f, tails = g.funs[i], g.tails[i]
    if length(tails) == 0
      output = length(l) == 1 ? apply(f, input) : apply(f, input[k])
      k += 1
    else
      input = length(tails) == 1 ? outputs[tails[1]] : map(id -> outputs[id], tails)
      output = apply(f, input)
    end
    outputs[i] = output
  end
  Variable(outputs[end])
end
