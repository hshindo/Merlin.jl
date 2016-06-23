using HDF5

"""
    save(dict, path)

Save Merlin objects as a HDF5 format.
Supported objects are `Graph` and `Var`.
"""
function save(dict::Dict, path)
  function write(g, key, val)
    if typeof(val) <: Dict
      g = g_create(g, string(key))
      for (k,v) in val
        write(g, k, v)
      end
    else
      g[string(key)] = val
    end
  end

  h5open(path, "w") do h
    g = g_create(h, "Merlin")
    for (k,v) in dict
      write(g, k, v)
    end
  end
end

function load(path)
  dict = h5read(path, "Merlin")
  for (k,v) in dict
    if typeof(v) <: Dict
      load(eval(parse(k)), v)
    end
  end
end

function hdf5dict(v::Var)
  d = Dict()
  d["value"] = typeof(v.value) == Symbol ? string(value) : v.value
  d["f"] = string(v.f)
  d["argtype"] = string(typeof(v.args))
  d["args"] = Int[v.args...]
  Dict("Var" => d)
end

function hdf5dict(g::Graph)
  d_nodes = Dict()
  for i = 1:length(g.nodes)
    d_nodes[string(i)] = hdf5dict(g.nodes[i])
  end
  d_sym2id = Dict()
  for (k,v) in g.sym2id
    d_sym2id[string(k)] = v
  end
  "Graph" => Dict("nodes" => d_nodes, "sym2id" => d_sym2id)
end

function load(::Type{Graph}, dict)
  nodes = Var[]
  dict = h5read(path, "graph")
  for (k,v) in dict["nodes"]
    id = parse(Int, k)
    while id > length(nodes)
      push!(nodes, Var(nothing))
    end
    nodes[id] = v
  end
end

function load(::Type{Var}, dict)

end

#=
function todict{T}(x::T)
  d = Dict()
  names = fieldnames(T)
  for i = 1:nfields(T)
    f = getfield(x, i)
    d[string(names[i])] = f
  end
  d
end
=#
