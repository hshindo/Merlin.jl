type Custom <: Functor
  f::Function
end

apply(fun::Custom, input) = fun.f(input)

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

function Base.|>(input::Tuple, g::Graph)
  outputs = Array(Any, length(g.funs))
  k = 1
  for i = 1:length(g.funs)
    f, tails = g.funs[i], g.tails[i]
    if length(tails) == 0
      outputs[i] = input[k] |> f
      k += 1
    elseif length(tails) == 1
      outputs[i] = outputs[tails[1]] |> f
    else
      inputs = map(id -> outputs[id], tails)
      outputs[i] = inputs |> f
    end
  end
  outputs[end]
end

function apply(g::Graph, vars::Vector{Variable})
  outputs = Array(Any, length(g.funs))
  k = 1
  for i = 1:length(g.funs)
    f, tails = g.funs[i], g.tails[i]
    if length(tails) == 0
      outputs[i] = vars[k] |> f
      k += 1
    elseif length(tails) == 1
      outputs[i] = outputs[tails[1]] |> f
    else
      input = map(id -> outputs[id], tails)
      outputs[i] = input |> f
    end
  end
  outputs[end]
end

function apply2(g::Graph, vars::Vector{Variable})
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
