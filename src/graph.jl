type MemoryPool
  buffer::Vector{Array}
  ref::Vector{Array}
  index::Int
end

function allocate{T}(mp::MemoryPool, ::Type{T}, dims::Int...)
  buffer, index = mp.buffer, mp.index
  if mp.index > length(mp.buffer)
    push!(mp.buffer, Array(T, dims))
  end
  a = buffer[index]
  if eltype(a) != T || size(a) != dims
    a = Array(T, dims)
  end
  a
end

free(mp::MemoryPool) = mp.index = 1

type Graph <: Functor
  funs::Vector{Functor}
  tails::Vector{Vector{Int}}
  mempools::Vector{MemoryPool}
  freemems::Vector
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

function forward(g::Graph, vars::Variable...)
  outputs = Array(Variable, length(g.funs))
  k = 1
  for i = 1:length(g.funs)
    f, tails = g.funs[i], g.tails[i]
    if length(tails) == 0
      outputs[i] = input[k] |> f
      k += 1
    elseif length(tails) == 1
      outputs[i] = outputs[tails[1]] |> f
    else
      vars = map(id -> outputs[id], tails)
      outputs[i] = vars |> f
    end
  end
  outputs[end]
end
