using HDF5
using ProgressMeter

function embeds2h5(path::String)
    keys = String[]
    values = Float32[]
    lines = open(readlines, path)
    prog = Progress(length(lines))
    for line in lines
        items = split(line, " ")
        push!(keys, items[1])
        for i = 2:length(items)
            push!(values, parse(Float32,items[i]))
        end
        next!(prog)
    end
    value = reshape(values, length(values)÷length(keys), length(keys))

    outpath = splitext(path)[1] * ".h5"
    println("Key size: $(length(keys))")
    println("Value size: $(size(value))")
    h5write(outpath, "key", keys)
    h5write(outpath, "value", value)
    println("Finished.")
end

if isempty(ARGS)
    println("julia embeds2h5.jl <filename>")
else
    embeds2h5(ARGS[1])
end
