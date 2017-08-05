function readdata(path::String, worddict::Dict, chardict::Dict, tagdict::Dict)
    unkwordid = worddict["UNKNOWN"]
    data_w, data_c, data_t = Var[], Var[], Var[]

    lines = open(readlines, path)
    i = 1
    while i <= length(lines)
        j = i
        while j < length(lines) && !isempty(lines[j+1])
            j += 1
        end
        w, cs, t = Int[], Vector{Int}[], Int[]
        for k = i:j
            items = split(lines[k], '\t')
            word, tag = items[2], items[5]
            word0 = replace(word, r"[0-9]", '0')
            wordid = get(worddict, lowercase(word0), unkwordid)
            chars = Vector{Char}(word0)
            charids = map(c -> get!(chardict,c,length(chardict)+1), chars)
            tagid = get!(tagdict, tag, length(tagdict)+1)
            push!(w, wordid)
            push!(cs, charids)
            push!(t, tagid)
        end
        push!(data_w, Var(w))
        batchdims = map(length, cs)
        c = cat(1, cs...)
        push!(data_c, Var(c,batchdims))
        push!(data_t, Var(t))
        i = j + 2
    end
    data_w, data_c, data_t
end
