function save(path::String, hdf5::HDF5Dict)
    function write(g, h5::HDF5Dict)
        for (k,v) in h5.dict
            if typeof(v) <: HDF5Dict
                gg = g_create(g, k)
                write(gg, v)
            else
                g[k] = v
            end
        end
    end

    h5open(path, "w") do h
        h["version"] = string(VERSION)
        g = g_create(h, "Merlin")
        write(g, hdf5)
    end
end
