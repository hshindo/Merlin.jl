using HDF5

export save_hdf5, load_hdf5, to_dict

"""
    save_hdf5(path, dict)

Save dictionary as a HDF5 format.
A key is stored as a group name, and a value is stored as a dataset in hdf5.
"""
function save_hdf5(path::String, dict::Dict)
    function _write(g, d::Dict)
        for (k,v) in d
            if typeof(v) <: Dict
                _write(g_create(g,string(k)), v)
            else
                g[string(k)] = v
            end
        end
    end
    h5open(path, "w") do h
        g = g_create(h, "Merlin")
        _write(g, dict)
    end
end

"""
    load_hdf5
"""
function load_hdf5(path::String)
    dict = h5read(path, "Merlin")
    for (k,v) in dict
        #if typeof(v) <: Dict
        #    load_hdf5(eval(parse(k)), v)
        #end
    end
    dict
end

function to_dict{T}(x::T)
    dict = Dict()
    names = fieldnames(T)
    for i = 1:nfields(T)
        f = getfield(x, i)
        dict[string(names[i])] = f
    end
    Dict(string(T) => dict)
end

function load_hdf5(::Type{Graph}, dict)
    nodes = Var[]
    for (k,v) in dict["nodes"]
        id = parse(Int, k)
        while id > length(nodes)
            push!(nodes, Var(nothing))
        end
        nodes[id] = v
    end
end

function load_hdf5(::Type{Graph}, path::String)
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
function to_hdf5(g::Graph)
d_nodes = Dict()
for i = 1:length(g.nodes)
d_nodes[string(i)] = to_hdf5(g.nodes[i])
end
d_sym2id = Dict()
for (k,v) in g.sym2id
d_sym2id[string(k)] = v
end
Dict("Graph" => Dict("nodes" => d_nodes, "sym2id" => d_sym2id))
end
=#
