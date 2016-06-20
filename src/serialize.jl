export save_hdf5

function save_hdf5(dict::Dict, path)
  function write_hdf5(g, key, val)
    if applicable(hdf5dict, val)
      d = hdf5dict(val)::Dict
      g = g_create(g, string(key))
      for (k,v) in d
        write_hdf5(g, k, v)
      end
    elseif typeof(val) <: Dict
      g = g_create(g, string(key))
      for (k,v) in val
        write_hdf5(g, k, v)
      end
    else
      g[string(key)] = val
    end
  end

  h5open(path, "w") do h
    g = g_create(h, "Merlin")
    for (k,v) in dict
      write_hdf5(g, k, v)
    end
  end
end

function hdf5dict(v::Var)
  d = Dict()
  d["value"] = typeof(v.value) == Symbol ? string(value) : v.value
  d["f"] = string(v.f)
  d["argtype"] = string(typeof(v.args))
  d["args"] = Int[v.args...]
  d
end

function hdf5dict(g::Graph)
  d_nodes = Dict()
  for i = 1:length(g.nodes)
    n = g.nodes[i]
    d_nodes[string(i)] = n
  end
  d_sym2id = Dict()
  for (k,v) in g.sym2id
    d_sym2id[string(k)] = v
  end
  Dict("nodes" => d_nodes, "sym2id" => d_sym2id)
end

function load(path)
  dict = h5read(path, "Merlin")
  
end

function load(::Type{Graph}, path)
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
