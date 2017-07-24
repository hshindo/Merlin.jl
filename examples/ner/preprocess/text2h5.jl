using HDF5

"""
Convert plain-text to HDF5 format.
This is used for converting plain-text word embeddings file into HDF5.
"""
function text2h5(path::String; delim=' ')
    T = Float32
    keys = String[]
    values = T[]
    lines = open(readlines, path)
    for line in lines
        items = split(line, delim)
        push!(keys, String(items[1]))
        value = [parse(T,items[i]) for i=2:length(items)]
        append!(values, value)
    end
    n = length(values) ÷ length(lines)
    if findfirst(x -> x == "UNKNOWN", keys) == 0
        push!(keys, "UNKNOWN")
        append!(values, zeros(T,n))
    end
    v = reshape(values, n, length(keys))
    h5write(path*".h5", "key", keys)
    h5write(path*".h5", "value", v)
end

path = joinpath(dirname(@__FILE__), "glove.840B.300d.txt")
text2h5(path)
