function readdata(path::String, worddict::Dict, chardict::Dict, tagdict::Dict)
    unkword = worddict["UNKNOWN"]
    data_w, data_c, data_t = Var[], Var[], Var[]
    w, c, t = Int[], Vector{Int}[], Int[]

    lines = open(readlines, path)
    for i = 1:length(lines)
        if isempty(lines[i])
            isempty(w) && continue
            push!(data_w, Var(w))
            push!(data_c, Var(c))
            push!(data_t, Var(t))
            w, c, t = Int[], Vector{Int}[], Int[]
        else
            items = split(lines[i], '\t')
            word, tag = items[2], items[5]
            word0 = replace(word, r"[0-9]", '0')
            wordid = get(worddict, lowercase(word0), unkword)
            chars = Vector{Char}(word0)
            charid = map(c -> get!(chardict,c,length(chardict)+1), chars)
            tagid = get!(tagdict, tag, length(tagdict)+1)
            push!(w, wordid)
            push!(c, charid)
            push!(t, tagid)
        end
    end
    if !isempty(w)
        push!(data_w, Var(w))
        push!(data_c, Var(c))
        push!(data_t, Var(t))
    end
    data_w, data_c, data_t
end
