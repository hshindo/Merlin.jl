function conv(path::String)
    data = []
    lines = open(readlines, path)
    lines = map(chomp, lines)
    for i = 1:length(lines)
        line = lines[i]
        if isempty(line)
            push!(data, "")
            continue
        end
        items = split(line, " ")
        word = String(items[1])
        tag = String(items[4])
        if startswith(tag, "I-")
            if i == 1 || isempty(lines[i-1])
                tag = replace(tag, "I-", "")
            else
                ptag = String(split(lines[i-1])[4])
                if ptag == "O"
                    tag = replace(tag, "I-", "")
                elseif startswith(ptag, "B-")
                    tag = "_"
                elseif startswith(ptag, "I-")
                    tag = "_"
                else
                    throw("$ptag")
                end
            end
        elseif startswith(tag, "B-")
        elseif tag == "O"
        else
            throw("Invalid tag: $tag.")
        end
        push!(data, "$(word)\t$(tag)")
    end
    open("a.out", "w") do f
        foreach(x -> println(f,x), data)
    end
end

function convIOBES(path::String)
    data = []
    lines = open(readlines, path)
    lines = map(chomp, lines)
    for i = 1:length(lines)
        line = lines[i]
        if isempty(line)
            push!(data, "")
            continue
        end
        items = split(line, "\t")
        word = String(items[1])
        tag = String(items[2])
        if startswith(tag, "B-") || startswith(tag, "S-")
            tag = tag[3:end]
        elseif startswith(tag, "I-") || startswith(tag, "E-")
            tag = "_"
        end
        push!(data, "$(word)\t$(tag)")
    end
    open("a.out", "w") do f
        foreach(x -> println(f,x), data)
    end
end

workspace()
convIOBES(joinpath(dirname(@__FILE__), ".data/eng.train.IOBES"))
