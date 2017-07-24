function convertBI(path::String)
    data = []
    for line in open(readlines,path)
        line = chomp(line)
        if isempty(line)
            push!(data, "")
        else
            items = split(line, ' ')
            tag = items[3]
            if tag == "O"
            elseif tag[1] == 'B'
                tag = tag[3:end]
            elseif tag[1] == 'I'
                tag = "_"
            else
                tbrow("")
            end
            push!(data, "$(items[1])\t$(items[2])\t$tag")
        end
    end
    open("test.out", "w") do f
        foreach(x -> println(f,x), data)
    end
end

convertBI("C:/Users/hshindo/Dropbox/chunking/test.txt")
