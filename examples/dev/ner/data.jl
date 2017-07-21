function aaa()
end

function readdata!(seg::Segmenter, path::String)
    data_w, data_c, data_t = Var[], Vector{Var}[], Var[]
    w, c, t = Int[], Var[], Int[]
    unkwordid = seg.word2id["UNKNOWN"]

    for line in open(readlines,path)
        if isempty(line)
            isempty(w) && continue
            push!(data_w, Var(w))
            push!(data_c, c)
            push!(data_t, Var(t))
            w, c, t = Int[], Var[], Int[]
        else
            items = split(line, "\t")
            word = String(items[1])
            #word = replace(word, r"[0-9]", '0')
            wordid = get(seg.word2id, lowercase(word), unkwordid)
            push!(w, wordid)

            chars = Vector{Char}(word)
            charids = map(chars) do c
                get!(seg.char2id, string(c), length(seg.char2id)+1)
            end
            push!(c, Var(charids))

            tag = String(items[2])
            tagid = get!(seg.tag2id, tag, length(seg.tag2id)+1)
            seg.id2tag[tagid] = tag
            push!(t, tagid)
        end
    end
    if !isempty(w)
        push!(data_w, Var(w))
        push!(data_c, c)
        push!(data_t, Var(t))
    end
    data_w, data_c, data_t
end
