"""
    save_hdf5

Save object as HDF5 format.
Object must be 
"""
function save_hdf5(dict::Dict, path)
  h5open(path, "w") do f
    root = g_create(f, "Merlin")
    for (k,v) in dict
      d = hdf5dict(v)
      g[string(k)] = hdf5
      for (kk,vv) in hdf5
        gg = g_create(root, string(k))
        #gg[]
      end

      g[string(k)] = to_hdf5(v)
      g = g_create(root, string(k))
      g[]
    end
  end
end

function to_hdf5(v::Var)
  value = v.value
  typeof(value) == Symbol && (value = string(value))
  f = string(v.f)
  argtype = string(typeof(v.args))
  args = Int[v.args...]
  "value" => value,
  "f" => f,
  "args" => args,
  "argtype" => argtype
end

function save(g::Graph, path)
  h5open(path, "w") do f
    g_root = g_create(h, "graph")
    g_nodes = g_create(g_root, "nodes")
    for i = 1:length(g.nodes)
      n = g.nodes[i]
      g_node = g_create(g_nodes, "$i")
      for (k,v) in to_dict(n)
        g_node[k] = v
      end
    end
    g_sym2id = g_create(g_root, "sym2id")
    for (k,v) in g.sym2id
      g_sym2id[string(k)] = v
    end
  end
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

function to_dict(v::Var)
  value = v.value
  typeof(value) == Symbol && (value = string(value))
  f = string(v.f)
  argtype = typeof(v.args) <: Tuple ? "tuple" : "vector"
  args = Int[v.args...]
  Dict(
    "value" => value,
    "f" => f,
    "args" => args,
    "argtype" => argtype)
end

function to_var(dict::Dict)
  dict["value"]
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
